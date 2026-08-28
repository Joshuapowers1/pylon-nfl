#!/usr/bin/env bash
# Rebuilds index.html from the template plus the current model outputs.
set -euo pipefail
cd "$(dirname "$0")"
python3 - <<'PY'
import pathlib
tpl   = pathlib.Path('app.template.html').read_text()
data  = pathlib.Path('../out/web_bundle.json').read_text().replace('</','<\\/')
logos = pathlib.Path('../data/logos/logos96.json').read_text().replace('</','<\\/')
out = tpl.replace('__LOGOS__', logos).replace('__DATA__', data)
pathlib.Path('index.html').write_text(out)
# docs/ is the GitHub Pages source, so the published site is always the file
# that was just built and tested rather than a copy someone forgot to update.
docs = pathlib.Path('../docs'); docs.mkdir(exist_ok=True)
(docs/'index.html').write_text(out)
(docs/'.nojekyll').write_text('')
print('index.html rebuilt:', round(len(out)/1024,1), 'KB  (also written to docs/ for Pages)')
PY
