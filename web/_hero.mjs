import { chromium } from 'playwright';
import fs from 'fs';
fs.writeFileSync('preview.html', `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>${fs.readFileSync('index.html','utf8')}</body></html>`);
const b = await chromium.launch({executablePath:'/opt/pw-browsers/chromium'});
for(const [t,w,h] of [['light',1440,620],['dark',1440,620],['dark',390,760]]){
  const p = await b.newPage({viewport:{width:w,height:h}, colorScheme:t, deviceScaleFactor:2});
  await p.goto('file://'+process.cwd()+'/preview.html');
  await p.waitForTimeout(800);
  await p.evaluate(()=>closeWelcome());
  await p.waitForTimeout(250);
  await p.screenshot({path:`hero-${t}-${w}.png`});
  await p.close();
}
await b.close();
