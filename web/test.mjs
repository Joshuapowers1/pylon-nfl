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
  page.on('console', m => { if(m.type()==='error') errors.push(`[${theme}] CONSOLE: ${m.text()}`); });
  await page.goto('file://'+process.cwd()+'/preview.html');
  await page.waitForTimeout(1200);
  for (const id of ['overview','teams','players','draft','market','sim','compare','model']) {
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

console.log(errors.length ? errors.join('\n') : 'CLEAN: no errors, all 7 pages rendered, both themes, no overflow');
