#!/usr/bin/env bash
# Pulls every raw input PYLON needs. Nothing in this repo is hand-typed.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw data/raw/polymarket

echo "play-by-play, 2017-2025..."
for y in 2017 2018 2019 2020 2021 2022 2023 2024 2025; do
  curl -sSL -o "data/raw/pbp_${y}.parquet" \
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_${y}.parquet" &
done
wait

echo "team reference table..."
curl -sSL -o data/raw/teams_nflverse.csv "https://raw.githubusercontent.com/nflverse/nfldata/master/data/teams.csv"

echo "schedules and 2026 rosters..."
curl -sSL -o data/raw/schedules.csv   "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
curl -sSL -o data/raw/roster_2026.csv "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_2026.csv"

echo "live PPR ADP..."
curl -sSL -o data/raw/adp_ppr_2026.json \
  "https://fantasyfootballcalculator.com/api/v1/adp/ppr?teams=12&year=2026&position=all"

echo "live Polymarket NFL book..."
curl -sSL -o data/raw/polymarket/events_nfl.json \
  "https://gamma-api.polymarket.com/events?limit=100&active=true&closed=false&tag_slug=nfl"

echo "team logos (fetched, not vendored -- see scripts/fetch_logos.py)..."
python3 scripts/fetch_logos.py

echo
echo "done. now run:"
echo "  PYTHONPATH=src python -m pylon.run --sims 25000 --player-sims 10000"
echo "  PYTHONPATH=src python -m pylon.webdata"
echo "  cd web && ./build.sh"
