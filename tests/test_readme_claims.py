"""The README quotes about forty numbers. This checks they are still true.

A projection repository lives or dies on whether its stated figures match the
figures it actually produces. Prose does not get regenerated when the model is
refit, so it drifts silently, and a reader who spot-checks one number and finds
it stale has no reason to trust the other thirty-nine.

This caught a real one: the de-vig table had a market snapshot from an earlier
build, with the two conference rows transposed on top of being stale.

Each claim names the section of the README it must appear in, so a number that
happens to occur elsewhere in the document cannot make a check pass by accident.
Claims are compared as formatted strings rather than parsed floats, because the
thing being verified is what a reader sees.
"""

from __future__ import annotations

import csv
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(rel: str):
    return json.loads((ROOT / rel).read_text())


def readme_sections() -> dict[str, str]:
    """Split the README on ## and ### headings, keyed by heading text."""
    text = (ROOT / "README.md").read_text()
    parts = re.split(r"^#{2,3} (.+)$", text, flags=re.MULTILINE)
    sections = {"": parts[0]}
    for name, body in zip(parts[1::2], parts[2::2]):
        sections[name.strip()] = body
    return sections


def num(x: float, places: int = 2) -> str:
    """Round the way a person writing prose rounds.

    Python's float formatting inherits binary representation error, so
    f"{18.725:.2f}" gives 18.72 while anyone writing the number by hand gives
    18.73. Comparing against what a reader sees means rounding half away from
    zero on the decimal value.
    """
    q = Decimal(1).scaleb(-places)
    return str(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def pct(x: float, places: int = 1) -> str:
    return f"{num(x * 100, places)}%"


def build_claims() -> list[tuple[str, str, str]]:
    """(section heading, expected string, what it describes)."""
    cal = load_json("config/calibration.json")
    bt = load_json("config/backtest.json")
    ins = load_json("config/inseason.json")
    gs = load_json("config/game_script.json")
    st = load_json("config/streaming_premium.json")
    led = load_json("out/prediction_ledger.json")

    shr, tr, var = cal["shrinkage"], cal["translation"], bt["variance_split"]
    claims: list[tuple[str, str, str]] = [
        ("The finding", num(shr['off_pass']['shrinkage'], 2), "offense pass carry-over"),
        ("The finding", num(shr['off_rush']['shrinkage'], 2), "offense rush carry-over"),
        ("The finding", num(shr['def_pass']['shrinkage'], 2), "defense pass carry-over"),
        ("The finding", num(shr['def_rush']['shrinkage'], 2), "defense rush carry-over"),

        ("The honest backtest", str(bt["n_team_seasons"]), "team-seasons tested"),
        ("The honest backtest", num(bt['mae_wins'], 2), "win-total MAE"),
        ("The honest backtest", num(bt['rmse_wins'], 2), "win-total RMSE"),
        ("The honest backtest", num(bt['correlation_pred_actual'], 2), "projected vs actual correlation"),
        ("The honest backtest", pct(bt["within_2_wins_pct"]), "within two wins"),
        ("The honest backtest", pct(bt["straight_up_game_accuracy"]), "straight-up accuracy"),

        ("In-season updating", pct(ins["frozen"]["accuracy"]), "frozen accuracy"),
        ("In-season updating", pct(ins["updated"]["accuracy"]), "updated accuracy"),
        ("In-season updating", num(ins['frozen']['mae'], 2), "frozen MAE"),
        ("In-season updating", num(ins['updated']['mae'], 2), "updated MAE"),
        ("In-season updating", num(ins['frozen']['logloss'], 3), "frozen log loss"),
        ("In-season updating", num(ins['updated']['logloss'], 3), "updated log loss"),
        ("In-season updating", str(ins["k_best"]), "fitted blend constant k"),

        ("4. Translate to points, calibrated on real games", num(tr['hfa'], 2), "home-field advantage"),
        ("4. Translate to points, calibrated on real games", num(tr['points_per_rating'], 2), "points per rating"),
        ("4. Translate to points, calibrated on real games", num(var['sigma_total_pts'], 2), "residual sigma"),

        ("5. Simulate, with the uncertainty that actually matters",
         num(var['sigma_game_pts'], 2), "game-level sigma"),
        ("5. Simulate, with the uncertainty that actually matters",
         num(var['sigma_team_pts'], 2), "team-constant sigma"),

        ("Weekly projections and start/sit", num(gs['rush_att_per_point'], 2), "rush attempts per point"),
        ("Weekly projections and start/sit", num(gs['pass_att_per_point'], 2), "pass attempts per point"),

        ("The number I got wrong", num(st['premium_ppg'], 1).lstrip("-"), "streaming premium"),
        ("The number I got wrong", str(st["replacement_rank"]), "replacement rank"),
    ]

    for season in st["by_season"]:
        for key, label in (("streamed_ppg", "streamed"), ("naive_ppg", "replacement-rank")):
            claims.append(("The number I got wrong",
                           num(season[key], 2),
                           f"{season['season']} {label} points per game"))

    # The de-vig table, the one that actually drifted.
    with (ROOT / "out" / "market_vig_report.csv").open() as fh:
        for row in csv.DictReader(fh):
            if int(row["n_outcomes"]) < 16:
                continue  # the division books are not in the README table
            sec = "7. De-vig before comparing anything"
            claims.append((sec, num(float(row["raw_sum"]), 4), f"{row['book']} raw sum"))
            claims.append((sec, num(abs(float(row["overround_pct"])), 2) + "%", f"{row['book']} overround"))

    claims.append(("The sealed ledger", led["digest"], "ledger digest"))
    claims.append(("The sealed ledger", led["stamped_at"], "ledger timestamp"))
    return claims


def test_readme_numbers_match_the_outputs():
    sections = readme_sections()
    missing = []
    for heading, expected, description in build_claims():
        body = sections.get(heading)
        if body is None:
            missing.append(f"  section {heading!r} not found in README (claim: {description})")
        elif expected not in body:
            missing.append(
                f"  {description}: outputs say {expected!r}, "
                f"not found under '## {heading}'"
            )
    assert not missing, (
        "README has drifted from the committed outputs:\n"
        + "\n".join(missing)
        + "\n\nRegenerate with `PYTHONPATH=src python -m pylon.run` and update the prose."
    )


def test_every_readme_screenshot_is_committed():
    """A README image that is gitignored renders as a broken icon on GitHub."""
    text = (ROOT / "README.md").read_text()
    refs = [m for m in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
            if not m.startswith("http")]
    assert refs, "no local images referenced; did the README change shape?"
    absent = [r for r in refs if not (ROOT / r).exists()]
    assert not absent, f"README references missing files: {absent}"


if __name__ == "__main__":
    for fn in (test_readme_numbers_match_the_outputs, test_every_readme_screenshot_is_committed):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            print(f"FAIL {fn.__name__}\n{exc}")
