#!/usr/bin/env bash
# Pulls every raw input PYLON needs. Nothing in this repo is hand-typed.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw data/raw/polymarket

# A completed season's play-by-play never changes again, so a file restored
# from the Actions cache is left alone. Without this the cache step in the
# weekly workflow buys nothing: curl would overwrite all 183MB every Tuesday
# regardless of what was already on disk.
echo "play-by-play, 2017-2025..."
for y in 2017 2018 2019 2020 2021 2022 2023 2024 2025; do
  f="data/raw/pbp_${y}.parquet"
  if [ -s "$f" ]; then echo "  ${y} already on disk, skipping"; continue; fi
  curl -sSL -o "$f" \
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_${y}.parquet" &
done
wait

# The season in progress is the one file that changes every week, so it is
# always re-downloaded. Before kickoff nflverse has not published it yet and the
# refit simply runs on the August model.
echo "play-by-play, 2026 to date..."
curl -sSfL -o data/raw/pbp_2026.parquet \
  "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2026.parquet" \
  || { echo "  no 2026 play-by-play published yet"; rm -f data/raw/pbp_2026.parquet; }

echo "team reference table..."
curl -sSL -o data/raw/teams_nflverse.csv "https://raw.githubusercontent.com/nflverse/nfldata/master/data/teams.csv"

echo "schedules and 2026 rosters..."
curl -sSL -o data/raw/schedules.csv   "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
curl -sSL -o data/raw/roster_2026.csv "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_2026.csv"

# ADP only means something before drafts happen. In season the feed thins to a
# few in-season mocks, so an existing preseason file is kept and the draft
# board stays frozen at its August build (see run.py).
echo "live PPR ADP..."
if [ -s data/raw/adp_ppr_2026.json ] && [ "${PYLON_REFRESH_ADP:-0}" != "1" ]; then
  echo "  keeping the ADP already on disk"
else
  curl -sSL -o data/raw/adp_ppr_2026.json \
    "https://fantasyfootballcalculator.com/api/v1/adp/ppr?teams=12&year=2026&position=all"
fi

echo "live Polymarket NFL book..."
curl -sSL -o data/raw/polymarket/events_nfl.json \
  "https://gamma-api.polymarket.com/events?limit=100&active=true&closed=false&tag_slug=nfl"

echo "team logos (fetched, not vendored -- see scripts/fetch_logos.py)..."
python3 scripts/fetch_logos.py

echo "live market snapshot (Polymarket books + this week's DraftKings lines)..."
python3 scripts/snapshot_markets.py

echo
echo "done. now run:"
echo "  PYTHONPATH=src python -m pylon.run --sims 25000 --player-sims 10000"
echo "  PYTHONPATH=src python -m pylon.webdata"
echo "  cd web && ./build.sh"
