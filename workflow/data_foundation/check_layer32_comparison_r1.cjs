const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {chromium}=require('/home/zjb/Desktop/worktrees/mfu-w37-v610/node_modules/playwright');
(async()=>{
const out='results/data-foundation/layer32-comparison-ui-r1';fs.mkdirSync(out,{recursive:true});
const browser=await chromium.launch({headless:true,args:['--disable-gpu','--no-zygote','--single-process'],executablePath:'/home/zjb/.cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell'});
const p=await browser.newPage({viewport:{width:1400,height:950}}),errors=[];p.on('pageerror',e=>errors.push(String(e)));
const url='http://192.168.8.16:43312/df-v001/layer-workload-r1/';if((await p.goto(url)).status()!==200)throw Error('http');
let checks=0;for(const mb of ['0','1','2','3'])for(const metric of ['active','retained'])for(const it of ['40','45','50','55','60','65','70','75','80']){
await p.selectOption('#source32-mb',mb);await p.selectOption('#source32-metric',metric);await p.selectOption('#source32-iter',it);
for(const [selector,count] of [['#source32-depth path',9],['#source32-heat rect',72],['#source32-compare path',3],['#source32-table tr',9],['#grid-heatmap rect',504]])if(await p.locator(selector).count()!==count)throw Error(selector);
checks++;}
await p.selectOption('#source32-mb','0');await p.selectOption('#source32-metric','active');await p.selectOption('#source32-iter','60');
const cells=await p.locator('#source32-table tr').nth(1).innerText();if(!cells.includes('147')||!cells.includes('59'))throw Error('known L3 values');
await p.locator('#source32-toggles button').first().click();if(await p.locator('#source32-depth path').count()!==8)throw Error('toggle');
await p.locator('#source32-comparison').screenshot({path:path.join(out,'section.png')});
await p.setViewportSize({width:390,height:844});if(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('overflow');
for(const name of ['source32-report.json','source32-manifest.json','source32-method.md'])if((await p.request.get(url+name)).status()!==200)throw Error(name);
await browser.close();if(errors.length)throw Error(errors.join('\n'));
const report={status:'PASS',control_combinations:checks,source_heatmap_cells:72,existing_grid_preserved:true,known_values_pass:true,mobile_overflow:false,errors,html_sha256:crypto.createHash('sha256').update(fs.readFileSync('results/data-foundation/2111-blocks-ui-r1/df-v001/layer-workload-r1/index.html')).digest('hex')};
fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n');console.log(report);
})().catch(e=>{console.error(e);process.exit(1)});
