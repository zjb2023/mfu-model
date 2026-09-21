const fs = require('fs');
const path = require('path');
const assert = require('assert');
const {chromium} = require('/home/zjb/Desktop/worktrees/mfu-w37-v610/node_modules/playwright');
const out = process.argv[2];
if (!out) throw Error('Provide new output directory');
fs.mkdirSync(out, {recursive:false});
(async()=>{
  const browser = await chromium.launch({headless:true,executablePath:'/home/zjb/.cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell'});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1000}});
    const errors=[]; page.on('pageerror',e=>errors.push(String(e)));
    const base='http://192.168.8.16:43312/df-v001/';
    const response=await page.goto(base+'prediction-overview-r1/',{waitUntil:'domcontentloaded'});
    assert.equal(response.status(),200);
    assert((await page.locator('#fb-error').innerText()).includes('6.16%'));
    assert((await page.locator('#overall').innerText()).includes('N/A'));
    assert.equal(await page.locator('#pipeline-chart .bar').count(),248);
    await page.locator('[aria-label="s1:F0"]').click();
    assert((await page.locator('#node-detail').innerText()).includes('342.714'));
    await page.selectOption('#mb','0');
    assert.equal(await page.locator('#pipeline-chart .bar').count(),62);
    await page.locator('#show-pp').uncheck();
    assert.equal(await page.locator('#pipeline-chart .bar').count(),32);
    await page.locator('#show-pp').check();await page.selectOption('#mb','all');
    const links={};
    for(const x of ['cp-ep8-structure-r1/','b-four-layers-r10/','four-layer-f-r1/','detailed-fb-32to256-r1/','pp32-detailed-fb-r1/']){
      links[x]=(await page.request.get(base+x)).status();assert.equal(links[x],200);
    }
    await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
    const disclosures=page.locator('details').filter({has:page.locator('iframe')});
    for(let i=0;i<2;i++){
      await disclosures.nth(i).locator('summary').click();
      const frame=await (await disclosures.nth(i).locator('iframe').elementHandle()).contentFrame();
      await frame.waitForSelector('h1');
      assert((await frame.locator('h1').innerText()).length>0);
      await disclosures.nth(i).locator('summary').click();
    }
    await page.setViewportSize({width:390,height:844});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify({status:'PASS',http:200,links,errors,all_nodes:248,filtered_nodes:62,fb_only_nodes:32,embedded_pages_loaded:2,mobile_overflow:false},null,2));
    console.log('PASS',out);
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
