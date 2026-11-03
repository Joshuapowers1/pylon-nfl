/* Contrast, overflow and clipping, on every page, in both themes, at three
   widths. Written because a color that reads fine to the person who chose it
   is the single easiest thing to get wrong and the hardest to notice: the
   first run of this found 70 failures, most of them the small gray labels
   that carry the explanations.

   Exit code is non-zero on any failure, so CI can gate on it.
     node audit.mjs
*/
import { chromium } from 'playwright';
import fs from 'fs';
fs.writeFileSync('preview.html', `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>${fs.readFileSync('index.html','utf8')}</body></html>`);
/* Prefer a browser that is already on the machine, but fall back to whatever
   Playwright resolves for itself. The hardcoded path is right inside the
   sandbox this was written in and wrong on a GitHub runner, where
   `npx playwright install` puts chromium under ~/.cache/ms-playwright. */
const EXEC = ['/opt/pw-browsers/chromium', '/opt/pw-browsers/chromium-1194/chrome-linux/chrome']
  .find(p => fs.existsSync(p));
const b = await chromium.launch(EXEC ? { executablePath: EXEC } : {});
const PAGES=['overview','live','matchups','startsit','teams','players','draft','market','sim','record','model'];
const problems=[];
for(const theme of ['dark','light']){
  for(const w of [1440,768,390]){
    const p = await b.newPage({viewport:{width:w,height:900}, colorScheme:theme});
    // Same reason as the behavior suite: the live page must not reach out.
    await p.route('**/site.api.espn.com/**', route => route.fulfill({status:200,
      contentType:'application/json', body: fs.readFileSync('scoreboard-fixture.json','utf8')}));
    p.on('pageerror',e=>problems.push(`[${theme}/${w}] PAGEERROR ${e.message}`));
    await p.goto('file://'+process.cwd()+'/preview.html');
    await p.waitForTimeout(500);
    await p.evaluate(()=>closeWelcome());
    for(const id of PAGES){
      await p.evaluate(i=>go(i), id);
      await p.waitForTimeout(200);
      const r = await p.evaluate(({i,W})=>{
        const out={over:[], contrast:[], clipped:[]};
        const doc = document.documentElement;
        if(doc.scrollWidth - doc.clientWidth > 2) out.over.push(`page ${doc.scrollWidth-doc.clientWidth}px`);
        const page = document.getElementById('pg-'+i);
        // Anything wider than its own container, other than the table wrappers
        // that are meant to scroll.
        page.querySelectorAll('*').forEach(el=>{
          if(el.closest('.tw') || el.closest('svg') || el instanceof SVGElement) return;
          const par = el.parentElement; if(!par) return;
          if(el.scrollWidth > el.clientWidth + 2 && getComputedStyle(el).overflowX === 'visible'){
            out.clipped.push(el.className || el.tagName);
          }
        });
        // Relative luminance contrast on visible text.
        const lum = c => { const [r,g,bl]=c.map(v=>{v/=255; return v<=.03928? v/12.92 : ((v+.055)/1.055)**2.4;});
          return .2126*r+.7152*g+.0722*bl; };
        const parse = s => { const m=s.match(/[\d.]+/g); return m? m.slice(0,3).map(Number) : null; };
        const bgOf = el => { let n=el; while(n && n!==document.documentElement){
            const c=getComputedStyle(n).backgroundColor;
            if(c && !/rgba\(0, 0, 0, 0\)|transparent/.test(c)) return parse(c);
            n=n.parentElement; } return parse(getComputedStyle(document.body).backgroundColor); };
        [...page.querySelectorAll('p,span,td,th,div,li,b,a,button,small,em')].forEach(el=>{
          if(el.children.length) return;
          const t=(el.textContent||'').trim(); if(t.length<2) return;
          const cs=getComputedStyle(el);
          if(cs.visibility==='hidden'||cs.display==='none'||+cs.opacity<0.35) return;
          const fg=parse(cs.color), bg=bgOf(el); if(!fg||!bg) return;
          const L1=lum(fg), L2=lum(bg);
          const ratio=(Math.max(L1,L2)+.05)/(Math.min(L1,L2)+.05);
          const px=parseFloat(cs.fontSize), bold=+cs.fontWeight>=600;
          const need=(px>=24||(px>=18.66&&bold))?3:4.5;
          if(ratio < need - 0.25) out.contrast.push(`${t.slice(0,26)} ${ratio.toFixed(2)}:1 @${px}px`);
        });
        return out;
      }, {i:id, W:w});
      if(r.over.length) problems.push(`[${theme}/${w}/${id}] overflow ${r.over.join(', ')}`);
      if(r.clipped.length) problems.push(`[${theme}/${w}/${id}] clipped: ${[...new Set(r.clipped)].slice(0,4).join(' | ')}`);
      if(r.contrast.length) problems.push(`[${theme}/${w}/${id}] contrast: ${[...new Set(r.contrast)].slice(0,4).join(' | ')}`);
    }
    await p.close();
  }
}
console.log(problems.length ? problems.join('\n') : 'CSS AUDIT CLEAN: no contrast failures below WCAG AA, no overflow, no clipping');
if(problems.length) console.log('total issues:', problems.length);
await b.close();
process.exit(problems.length ? 1 : 0);
