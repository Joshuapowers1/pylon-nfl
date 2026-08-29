import { chromium } from 'playwright';
import fs from 'fs';

const body = fs.readFileSync('index.html','utf8');
fs.writeFileSync('preview.html', `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>${body}</body></html>`);

const browser = await chromium.launch({ executablePath: process.env.PW_CHROME });
const errors = [];
for (const theme of ['dark','light']) {
  const page = await browser.newPage({ viewport:{width:1440,height:1000},
    colorScheme: theme });
  page.on('pageerror', e => errors.push(`[${theme}] PAGEERROR: ${e.message}`));
  // The webfont is the one external request on the page and it is meant to
  // fall back silently, so a sandbox with no network is not a test failure.
  page.on('console', m => { if(m.type()==='error' && !/Failed to load resource/.test(m.text()))
    errors.push(`[${theme}] CONSOLE: ${m.text()}`); });
  await page.goto('file://'+process.cwd()+'/preview.html');
  await page.waitForTimeout(1200);
  for (const id of ['overview','myteam','matchups','startsit','teams','players','draft','market','sim','record','model']) {
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
/* Roster import. Both routes matter and neither can be checked by eye:
   the paste parser has to survive whatever an app puts around a name, and
   the Sleeper route has to map ids onto our own rows. The transport is
   stubbed with a response shaped exactly like the live API so this runs in
   CI with no network. */
const r = await browser.newPage({ viewport:{width:1440,height:1400}, colorScheme:'dark' });
r.on('pageerror', e=>errors.push(`[import] PAGEERROR: ${e.message}`));
await r.goto('file://'+process.cwd()+'/preview.html');
await r.waitForTimeout(900);
await r.evaluate(()=>{ try{localStorage.removeItem('pylon-myteam-v1')}catch(e){} go('myteam'); });
await r.waitForTimeout(300);

// A roster copied out of ESPN, position prefixes, bye weeks, injury flags,
// a defense that will never match, and a name that does not exist.
await r.click('#mtSrc button[data-s="paste"]');
await r.fill('#mtPaste', [
  "QB  Josh Allen Buf QB1  Bye 7",
  "RB  Bijan Robinson Atl RB2",
  "RB  De'Von Achane Mia RB4   Q",
  "WR  Ja'Marr Chase Cin WR1",
  "WR  Malik Nabers NYG WR9",
  "WR  Garrett Wilson NYJ WR14",
  "TE  Brock Bowers LV TE1",
  "FLEX Chase Brown Cin RB21",
  "BE  Jayden Daniels Was QB6",
  "BE  Tyrone Tracy Jr. NYG RB33",
  "BE  Jaxon Smith-Njigba Sea WR6",
  "BE  Trey McBride Ari TE3",
  "D/ST Ravens D/ST",
].join('\n'));
await r.click('#mtParse');
await r.waitForTimeout(600);
const pasted = await r.evaluate(()=>({
  n: MT.roster.length,
  chase: MT.roster.includes("Ja'Marr Chase"),
  tracy: MT.roster.includes("Tyrone Tracy Jr."),
  filled: [...document.querySelectorAll('#pg-myteam .slot')].filter(s=>!s.classList.contains('empty')).length,
  pts: document.querySelector('#pg-myteam .kpi .v')?.textContent,
}));
if(pasted.n !== 12) errors.push(`[import] paste matched ${pasted.n} players, expected 12`);
if(!pasted.chase) errors.push('[import] apostrophe name did not resolve');
if(!pasted.tracy) errors.push('[import] suffixed name did not resolve');
if(pasted.filled < 8) errors.push(`[import] only ${pasted.filled} lineup slots filled`);
if(!(parseFloat(pasted.pts) > 50)) errors.push(`[import] lineup total read ${pasted.pts}`);
await r.screenshot({path:'shot-myteam.png'});

// Sleeper, with the transport stubbed. Ids below are real Sleeper ids for
// the three players named, which is the point of the check.
await r.evaluate(()=>{
  const ids = Object.entries(IDS.sleeper);
  const idOf = n => (ids.find(([,v])=>v[0]===n)||[null])[0];
  const mine  = ['Josh Allen','Bijan Robinson','Brock Bowers','Puka Nacua','Chase Brown'].map(idOf);
  const other = ['Saquon Barkley','CeeDee Lamb','Trey McBride'].map(idOf);
  window.__stubIds = {mine, other};
  window.fetch = async (url)=>{
    const u = String(url);
    let body;
    if(/\/v1\/user\/[^/]+$/.test(u)) body = {user_id:'42', display_name:'test'};
    else if(/\/leagues\/nfl\//.test(u)) body = [{league_id:'99', name:'Test League', total_rosters:12,
      roster_positions:['QB','RB','RB','WR','WR','TE','FLEX','BN','BN','K','DEF']}];
    else if(/league\/99\/rosters/.test(u)) body = [
      {roster_id:1, owner_id:'42', players:mine},
      {roster_id:2, owner_id:'43', players:other}];
    else return {ok:false, status:404, json:async()=>({})};
    return {ok:true, status:200, json:async()=>body};
  };
});
await r.click('#mtSrc button[data-s="sleeper"]');
await r.fill('#slUser','test');
await r.click('#slGo');
await r.waitForTimeout(500);
await r.click('#slLoad');
await r.waitForTimeout(800);
const sl = await r.evaluate(()=>({
  ids: window.__stubIds,
  mine: MT.roster, taken: MT.taken.length, teams: MT.teams, slots: MT.slots,
  adds: [...document.querySelectorAll('#mtAdds tbody tr td:first-child')].map(td=>td.textContent.trim()),
}));
if(sl.ids.mine.some(x=>!x)) errors.push('[import] Sleeper id table is missing a named player');
if(sl.mine.length !== 5) errors.push(`[import] Sleeper roster resolved ${sl.mine.length} of 5`);
if(sl.taken !== 8) errors.push(`[import] league-wide rostered count was ${sl.taken}, expected 8`);
if(String(sl.slots) !== 'QB,RB,RB,WR,WR,TE,FLEX') errors.push(`[import] league slots read ${sl.slots}`);
if(sl.adds.some(n=>sl.mine.includes(n))) errors.push('[import] waiver list is offering a player already rostered');
if(sl.adds.includes('Trey McBride')) errors.push('[import] waiver list is offering another team\'s player');
await r.close();

// mobile
const m = await browser.newPage({ viewport:{width:390,height:844}, colorScheme:'dark' });
m.on('pageerror', e=>errors.push(`[mobile] PAGEERROR: ${e.message}`));
await m.goto('file://'+process.cwd()+'/preview.html');
await m.waitForTimeout(900);
await m.evaluate(()=>go('teams')); await m.waitForTimeout(400);
await m.screenshot({path:'shot-mobile.png'});
const mow = await m.evaluate(()=>document.documentElement.scrollWidth - document.documentElement.clientWidth);
if(mow>2) errors.push(`[mobile] horizontal overflow of ${mow}px at 390`);
await browser.close();

console.log(errors.length ? errors.join('\n') : 'CLEAN: 11 pages in both themes, roster import from paste and Sleeper, no overflow at 1440 or 390');
