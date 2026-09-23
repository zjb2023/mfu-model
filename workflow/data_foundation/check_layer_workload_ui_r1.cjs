const fs=require('fs');
const path=require('path');
const {chromium}=require('/home/zjb/Desktop/worktrees/mfu-w37-v610/node_modules/playwright');
(async()=>{
const out=path.resolve('results/data-foundation/layer-workload-ui-checks-r1');fs.mkdirSync(out);
const b=await chromium.launch({headless:true,timeout:120000,args:['--disable-gpu','--no-zygote','--single-process'],executablePath:'/home/zjb/.cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell'});
const p=await b.newPage({viewport:{width:1360,height:900}}),errors=[];p.on('pageerror',e=>errors.push(String(e)));
const url='http://192.168.8.16:43312/df-v001/layer-workload-r1/';let response=await p.goto(url);if(response.status()!==200)throw Error('HTTP');let count=0;
for(const w of ['224','256']){await p.selectOption('#world',w);for(let m=0;m<(w==='224'?3:4);m++){await p.selectOption('#mb',String(m));for(const phase of ['F','recompute','backward','B']){await p.selectOption('#phase',phase);if(await p.locator('svg').count()!==6)throw Error('missing chart');if(await p.locator('svg circle').count()<100)throw Error('missing points');count++;}}}
await p.screenshot({path:path.join(out,'desktop.png'),fullPage:true});await p.setViewportSize({width:390,height:844});if(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('mobile overflow');await p.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
for(const name of ['analysis.md','manifest.json']){if((await p.request.get(url+name)).status()!==200)throw Error(name)}
await p.goto('http://192.168.8.16:43312/df-v001/prediction-overview-r2/#b-evidence');if(await p.locator('a[href="../layer-workload-r1/"]').count()!==1)throw Error('overview link');
await b.close();if(errors.length)throw Error(errors.join('\n'));const result={status:'PASS',combinations:count,charts:6,http:200,mobileOverflow:false,overviewLink:true,errors};fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(result,null,2)+'\n');console.log(result);
})().catch(e=>{console.error(e);process.exit(1)});
