import { chromium } from 'playwright';
import fs from 'fs';

const body = fs.readFileSync('index.html','utf8');
fs.writeFileSync('preview.html', `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>${body}</body></html>`);

const browser = await chromium.launch({ executablePath: process.env.PW_CHROME });
const errors = [];

/* The live page calls a real endpoint. Under file:// the origin is null, so the
   browser refuses it before any of our code runs and logs a CORS error that no
   catch can suppress. Intercepting at the network layer instead of stubbing
   window.fetch satisfies the browser, keeps the suite offline, and exercises
   the same parsing path a real response would. */
const SCOREBOARD = JSON.parse(fs.readFileSync('scoreboard-fixture.json','utf8'));
const stubScoreboard = pg => pg.route('**/site.api.espn.com/**', route =>
  route.fulfill({status:200, contentType:'application/json', body: JSON.stringify(SCOREBOARD)}));
for (const theme of ['dark','light']) {
  const page = await browser.newPage({ viewport:{width:1440,height:1000},
    colorScheme: theme });
  page.on('pageerror', e => errors.push(`[${theme}] PAGEERROR: ${e.message}`));
  // The webfont is the one external request on the page and it is meant to
  // fall back silently, so a sandbox with no network is not a test failure.
  page.on('console', m => { if(m.type()==='error' && !/Failed to load resource/.test(m.text()))
    errors.push(`[${theme}] CONSOLE: ${m.text()}`); });
  await stubScoreboard(page);
  await page.goto('file://'+process.cwd()+'/preview.html');
  await page.waitForTimeout(1200);
  if(theme==='dark'){
    // First visit: the overlay should be up, and dismissing it should stick.
    const up = await page.evaluate(()=>document.getElementById('welcome').classList.contains('on'));
    if(!up) errors.push('[first visit] welcome overlay did not appear');
    await page.screenshot({path:'shot-welcome.png'});
  }
  await page.evaluate(()=>closeWelcome());
  await page.waitForTimeout(150);
  for (const id of ['overview','live','matchups','startsit','teams','players','draft','market','sim','record','model']) {
    await page.evaluate(i => go(i), id);
    await page.waitForTimeout(500);
    const vis = await page.evaluate(i => {
      const s=document.getElementById('pg-'+i);
      return { on: s.classList.contains('on'), len: s.innerHTML.length };
    }, id);
    if(!vis.on || vis.len < 400) errors.push(`[${theme}] page ${id} rendered ${vis.len} chars`);
    if(theme==='dark') await page.screenshot({ path:`shot-${id}.png`, fullPage:false });
  }
  // exercise the sheets
  await page.evaluate(()=>{ go('teams'); teamSheet('SEA'); });
  await page.waitForTimeout(400);
  if(theme==='dark') await page.screenshot({ path:'shot-team-sheet.png' });
  await page.evaluate(()=>{ closeSheet(); const p=PLAYERS.find(x=>x.player==='Josh Allen'); playerSheet(p); });
  await page.waitForTimeout(400);
  if(theme==='dark') await page.screenshot({ path:'shot-player-sheet.png' });
  await page.evaluate(()=>{ closeSheet(); openPal(); });
  await page.waitForTimeout(200);
  await page.fill('#palQ','mcbride'); await page.waitForTimeout(250);
  const hits = await page.evaluate(()=>palHits.length);
  if(hits<1) errors.push(`[${theme}] palette search found nothing for "mcbride"`);
  if(theme==='dark') await page.screenshot({ path:'shot-palette.png' });
  await page.evaluate(()=>closePal());
  // horizontal overflow check
  const ow = await page.evaluate(()=>document.documentElement.scrollWidth - document.documentElement.clientWidth);
  if(ow > 2) errors.push(`[${theme}] horizontal overflow of ${ow}px at 1440`);
  await page.close();
}
/* ------------------------------------------------------------------ */
/* The roster import (Sleeper, public ESPN, paste) was retired. Its checks went
   with it rather than being left to rot against elements that no longer exist.
   The live scoreboard that replaced it is covered below. */
const r = await browser.newPage({ viewport:{width:1440,height:1400}, colorScheme:'dark' });
r.on('pageerror', e=>errors.push(`[import] PAGEERROR: ${e.message}`));
await stubScoreboard(r);
await r.goto('file://'+process.cwd()+'/preview.html');
await r.waitForTimeout(900);
await r.evaluate(()=>closeWelcome());

/* Live scoreboard. The network is stubbed with a payload shaped exactly like
   ESPN's so this runs in CI with no connection, the same way the Sleeper route
   used to be checked. What matters is that every game gets a model number and
   a sentence a person can read, not that the feed is reachable. */
await r.evaluate(()=>go('live'));
await r.waitForTimeout(1500);
const lv = await r.evaluate(()=>({
  cards: document.querySelectorAll('#liveBody .livecard').length,
  says: document.querySelectorAll('#liveBody .live-say').length,
  blank: [...document.querySelectorAll('#liveBody .live-say')].filter(n=>!n.innerText.trim()).length,
  notice: !!document.querySelector('#liveBody .notice'),
}));
if(lv.notice) errors.push('[live] fell back to the offline notice with a good payload');
if(lv.cards < 12) errors.push(`[live] only ${lv.cards} games rendered`);
if(lv.says !== lv.cards) errors.push(`[live] ${lv.cards} cards but ${lv.says} explanations`);
if(lv.blank) errors.push(`[live] ${lv.blank} games rendered an empty sentence`);

// The model must price a game the pipeline never ran, from committed ratings.
const mm = await r.evaluate(()=>{
  const g = LIVE.games[0];
  return {pair:[g.away,g.home], m: modelMargin(g.away,g.home), p: winProb(modelMargin(g.away,g.home))};
});
if(!Number.isFinite(mm.m)) errors.push(`[live] no model margin for ${mm.pair.join(' at ')}`);
if(!(mm.p > 0 && mm.p < 1)) errors.push(`[live] win probability out of range: ${mm.p}`);
await r.screenshot({path:'shot-live.png'});

// The glossary. Every abbreviation in a column header should resolve to a
// definition, and clicking one should open the list at that entry.
await r.evaluate(()=>go('players'));
await r.waitForTimeout(500);
const gl = await r.evaluate(()=>{
  const heads = [...document.querySelectorAll('#pg-players thead th')].map(t=>t.textContent.trim());
  const wired = [...document.querySelectorAll('#pg-players thead th .term')].map(t=>t.textContent.trim());
  return {heads, wired, terms: TERMS.length};
});
if(gl.wired.length < 5) errors.push(`[glossary] only ${gl.wired.length} of ${gl.heads.length} player columns explain themselves`);
await r.evaluate(()=>document.querySelector('#pg-players thead th .term').click());
await r.waitForTimeout(400);
const opened = await r.evaluate(()=>({
  on: SHEET.classList.contains('on'),
  hit: !!SHEET.querySelector('.gl-row.hit'),
  rows: SHEET.querySelectorAll('.gl-row').length,
}));
if(!opened.on) errors.push('[glossary] clicking a column header did not open the glossary');
if(!opened.hit) errors.push('[glossary] the glossary opened but did not highlight the term clicked');
if(opened.rows !== gl.terms) errors.push(`[glossary] rendered ${opened.rows} of ${gl.terms} definitions`);
await r.evaluate(()=>{ document.getElementById('glQ').value='vig';
  document.getElementById('glQ').dispatchEvent(new Event('input')); });
await r.waitForTimeout(250);
const searched = await r.evaluate(()=>SHEET.querySelectorAll('.gl-row').length);
if(!(searched > 0 && searched < gl.terms)) errors.push(`[glossary] search for "vig" returned ${searched} of ${gl.terms}`);
await r.screenshot({path:'shot-glossary.png'});
// Sorting must not strip the affordance back off the header.
await r.evaluate(()=>closeSheet());
await r.waitForTimeout(250);
await r.evaluate(()=>{ const th=document.querySelector('#pg-players thead th:nth-child(5)'); th.click(); th.click(); });
await r.waitForTimeout(300);
const afterSort = await r.evaluate(()=>document.querySelectorAll('#pg-players thead th .term').length);
if(afterSort !== gl.wired.length) errors.push(`[glossary] sorting dropped the definition link: ${gl.wired.length} then ${afterSort}`);
await r.close();

// mobile
const m = await browser.newPage({ viewport:{width:390,height:844}, colorScheme:'dark' });
m.on('pageerror', e=>errors.push(`[mobile] PAGEERROR: ${e.message}`));
await stubScoreboard(m);
await m.goto('file://'+process.cwd()+'/preview.html');
await m.waitForTimeout(900);
await m.evaluate(()=>closeWelcome());
await m.evaluate(()=>go('teams')); await m.waitForTimeout(400);
await m.screenshot({path:'shot-mobile.png'});
const mow = await m.evaluate(()=>document.documentElement.scrollWidth - document.documentElement.clientWidth);
if(mow>2) errors.push(`[mobile] horizontal overflow of ${mow}px at 390`);
await browser.close();

console.log(errors.length ? errors.join('\n') : 'CLEAN: 11 pages in both themes, first-visit overlay, glossary wired to column headers, live scoreboard priced from committed ratings, no overflow at 1440 or 390');
