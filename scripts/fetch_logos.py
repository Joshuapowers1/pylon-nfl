"""
Fetch team logos and pack them into the data URI bundle the web app inlines.

Logos are fetched at build time rather than vendored into the repository. They
are trademarks of their respective clubs and this project has no licence to
redistribute them, only to display them for identification. Anyone building
from source pulls their own copy.

    python scripts/fetch_logos.py

Writes data/logos/logos96.json.

--------------------------------------------------------------------------
PYLON · NFL projection and market-edge engine
Joshua Powers · joshpowersbiz@gmail.com
github.com/Joshuapowers1/pylon-nfl · linkedin.com/in/joshuampowers · @joshuampowers
"""
from __future__ import annotations
import base64, io, json, sys, urllib.request
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "logos" / "logos96.json"
SRC = "https://a.espncdn.com/i/teamlogos/nfl/500/{slug}.png"
# nflverse and the logo CDN disagree about a handful of abbreviations.
ALIAS = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX", "AZ": "ARI"}
SIZE = 96          # renders crisply at the 44px crest on a 2x display


def main() -> int:
    teams = pd.read_csv(ROOT / "data" / "raw" / "teams_nflverse.csv")
    teams = teams[teams.season == teams.season.max()][["team", "espn"]].drop_duplicates("team")

    out, failed = {}, []
    for _, r in teams.iterrows():
        code = ALIAS.get(r["team"], r["team"])
        try:
            raw = urllib.request.urlopen(SRC.format(slug=str(r["espn"]).lower()), timeout=25).read()
        except Exception as e:                       # noqa: BLE001
            failed.append((code, str(e)))
            continue
        im = Image.open(io.BytesIO(raw)).convert("RGBA")
        # Trim transparent padding, then re-centre on a square canvas, so every
        # crest optically fills its box by the same amount. Without this the
        # wordmark-style logos look half the size of the round ones.
        bbox = im.split()[-1].getbbox()
        if bbox:
            im = im.crop(bbox)
        w, h = im.size
        side = max(w, h)
        canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        canvas.paste(im, ((side - w) // 2, (side - h) // 2))
        canvas = canvas.resize((SIZE, SIZE), Image.LANCZOS)
        buf = io.BytesIO()
        canvas.save(buf, "PNG", optimize=True)
        out[code] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"{len(out)} logos -> {OUT.relative_to(ROOT)} ({OUT.stat().st_size/1024:.0f} KB)")
    if failed:
        print("failed:", failed, file=sys.stderr)
    return 0 if len(out) == 32 else 1


if __name__ == "__main__":
    raise SystemExit(main())
