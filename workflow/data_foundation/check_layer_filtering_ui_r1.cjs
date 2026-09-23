const fs=require('fs'),path=require('path');
const {chromium}=require('/home/zjb/Desktop/worktrees/mfu-w37-v610/node_modules/playwright');
(async()=>{
const out=path.resolve('results/data-foundation/layer-filtering-ui-r1');fs.mkdirSync(out);
const browser=await chromium.launch({headless:true,timeout:120000,args:['--disable-gpu','--no-zygote','--single-process'],executablePath:'/home/zjb/.cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell'});
const page=await browser.newPage({viewport:{width:1360,height:900}}),errors=[];page.on('pageerror',e=>errors.push(String(e)));
const url='http://192.168.8.16:43312/df-v001/layer-workload-r1/';if((await page.goto(url)).status()!==200)throw Error('HTTP');
let checks=0;for(const w of ['224','256']){await page.selectOption('#world',w);for(let m=0;m<(w==='224'?3:4);m++){await page.selectOption('#mb',String(m));for(const phase of ['F','recompute','backward','B']){await page.selectOption('#phase',phase);if(await page.locator('svg').count()!==7)throw Error('charts');const text=await page.locator('#filtercomparison').innerText();if(!text.includes('当前MB'+m))throw Error('stale MB');checks++;}}}
const paths=await page.locator('#filtercurve svg path[fill="none"]').evaluateAll(es=>es.map(e=>e.getAttribute('d')));if(paths.length!==2||paths.some(d=>(d.match(/M/g)||[]).length!==14))throw Error('stage boundaries');
await page.locator('#filtering').screenshot({path:path.join(out,'filtering.png')});await page.setViewportSize({width:390,height:844});if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('mobile overflow');
for(const name of ['filtering.md','filtering-report.json','filtering-manifest.json'])if((await page.request.get(url+name)).status()!==200)throw Error(name);
await browser.close();if(errors.length)throw Error(errors.join('\n'));const report={status:'PASS',combinations:checks,charts:7,stage_segments:14,http:200,mobileOverflow:false,errors};fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n');console.log(report);
})().catch(e=>{console.error(e);process.exit(1)});
