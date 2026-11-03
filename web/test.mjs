import { chromium } from 'playwright';
import fs from 'fs';

const body = fs.readFileSync('index.html','utf8');
fs.writeFileSync('preview.html', `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>${body}</body></html>`);

/* Prefer a browser already on the machine, then the env override, then let
   Playwright resolve its own. The bare env var alone fails on a GitHub runner,
   where `npx playwright install` puts chromium under ~/.cache/ms-playwright. */
const EXEC = [process.env.PW_CHROME, '/opt/pw-browsers/chromium',
              '/opt/pw-browsers/chromium-1194/chrome-linux/chrome']
  .find(p => p && fs.existsSync(p));
const browser = await chromium.launch(EXEC ? { executablePath: EXEC } : {});
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

/* Every clock reads Eastern, labelled, whatever zone the reader is in.

   This is a regression test for a real bug. The schedule baked into the page
   comes from nflverse, whose gametime column is already Eastern with no zone
   marker on it; the page used to append a Z, declaring Eastern to be UTC, and
   then render the result in the reader's own zone. An 8:15 PM kickoff came out
   as 4:15 PM and the Sunday one o'clock slate came out as nine in the morning.
   Two things are checked and both matter: that no time is ever printed without
   ET after it, and that the labels are IDENTICAL from four different time
   zones, which is the property a zone-dependent formatter cannot have. */
const zoneLabels = {};
/* Both feeds matter and they are different code paths. With the scoreboard
   answering, kickoff labels are drawn from the event's UTC stamp; with it
   blocked - a locked-down browser, an offline laptop, an embedded preview -
   they are drawn from the schedule committed inside the page, and it was that
   second path that carried the four-hour shift. Checking only the first one
   would have let this bug through. */
for(const [tz, feed] of [['America/New_York','up'], ['America/New_York','blocked'],
                         ['America/Los_Angeles','up'], ['America/Los_Angeles','blocked'],
                         ['Asia/Tokyo','up'], ['Asia/Tokyo','blocked'],
                         ['UTC','up'], ['UTC','blocked']]){
  const c = await browser.newContext({timezoneId: tz, locale: 'en-US'});
  const z = await c.newPage();
  if(feed === 'up') await stubScoreboard(z);
  else await z.route('**/site.api.espn.com/**', route => route.abort());
  await z.goto('file://'+process.cwd()+'/preview.html');
  await z.waitForTimeout(700);
  await z.evaluate(()=>{ closeWelcome(); go('live'); });
  await z.waitForTimeout(1400);
  const labels = await z.$$eval('#liveBody .livecard .bar .dim', n=>n.map(x=>x.textContent.trim()));
  await z.evaluate(()=>go('matchups'));
  await z.waitForTimeout(700);
  const eyebrows = await z.$$eval('#mkGrid .mk .eyebrow', n=>n.map(x=>x.textContent.trim()));
  const all = labels.concat(eyebrows);
  const unlabelled = all.filter(t => /\d:\d\d\s?(AM|PM)/.test(t) && !/\bET\b/.test(t));
  const where = `${tz}, feed ${feed}`;
  if(!labels.length) errors.push(`[time] ${where}: the live page printed no kickoff times at all`);
  if(unlabelled.length) errors.push(`[time] ${where}: time with no zone on it: ${JSON.stringify(unlabelled.slice(0,3))}`);
  zoneLabels[where] = labels.join('|');
  await c.close();
}
/* Compared within a feed, not across them. The stubbed scoreboard is a fixed
   fixture from one week and the offline path reads whatever week the bundle is
   on, so from the next refit onward the two slates are different games. What
   must hold is that the reader's zone never changes a label. */
for(const feed of ['up', 'blocked']){
  const zoneKeys = Object.keys(zoneLabels).filter(k => k.endsWith('feed '+feed));
  const drifted = zoneKeys.filter(k => zoneLabels[k] !== zoneLabels[zoneKeys[0]]);
  if(drifted.length)
    errors.push(`[time] kickoff labels are not the same everywhere: ${drifted.join('; ')} differ from ${zoneKeys[0]}`);
}


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

console.log(errors.length ? errors.join('\n') : 'CLEAN: 11 pages in both themes, first-visit overlay, glossary wired to column headers, live scoreboard priced from committed ratings, every clock in ET from four reader time zones on both the live and the offline feed, no overflow at 1440 or 390');
