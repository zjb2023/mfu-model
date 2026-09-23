const fs=require('fs'),path=require('path'),crypto=require('crypto');
const{chromium}=require('/home/zjb/Desktop/worktrees/mfu-w37-v610/node_modules/playwright');
(async()=>{
const out=path.resolve('results/data-foundation/layer-iteration-ui-r1');fs.mkdirSync(out);
const b=await chromium.launch({headless:true,timeout:120000,args:['--disable-gpu','--no-zygote','--single-process'],executablePath:'/home/zjb/.cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell'});
const p=await b.newPage({viewport:{width:1360,height:900}}),errors=[];p.on('pageerror',e=>errors.push(String(e)));
if((await p.goto('http://192.168.8.16:43312/df-v001/layer-workload-r1/')).status()!==200)throw Error('HTTP');
const colors=['#00877a','#ad4b20','#6557a0'];let checks=0;
for(const world of ['224','256']){
 await p.selectOption('#world',world);
 const snapshot=await p.evaluate(()=>['total','maxcost','cv','filtercurve','pair-summary'].map(id=>document.getElementById(id).innerHTML));
 for(let mask=0;mask<8;mask++){
  for(let i=0;i<3;i++){const button=p.locator(`button[data-iteration="${[40,60,80][i]}"]`);if((await button.getAttribute('aria-pressed')==='true')!==!!(mask&(1<<i)))await button.click();}
  const expected=colors.filter((_,i)=>mask&(1<<i));
  for(const id of ['work','cost','active']){
   const lines=await p.locator(`#${id} path[fill="none"]`).evaluateAll(es=>es.map(e=>({d:e.getAttribute('d'),color:e.getAttribute('stroke')})));
   if(JSON.stringify(lines.map(x=>x.color))!==JSON.stringify(expected))throw Error('visibility/colors');
   for(const line of lines)if((line.d.match(/M/g)||[]).length!==1||(line.d.match(/L/g)||[]).length!==(world==='256'?55:47))throw Error('not continuous');
   if(mask===0&&!(await p.locator('#'+id).innerText()).includes('请选择至少一个iter'))throw Error('empty state');
  }
  const after=await p.evaluate(()=>['total','maxcost','cv','filtercurve','pair-summary'].map(id=>document.getElementById(id).innerHTML));if(JSON.stringify(after)!==JSON.stringify(snapshot))throw Error('sections2/4 changed');checks++;
 }
}
await p.locator('button[data-iteration="40"]').click();
for(const mb of ['0','1','2','3']){await p.selectOption('#mb',mb);for(const phase of ['F','recompute','backward','B']){await p.selectOption('#phase',phase);if(await p.locator('#work path[fill="none"]').count()!==2)throw Error('lost visibility state');checks++;}}
await p.locator('#work').screenshot({path:path.join(out,'continuous.png')});await p.setViewportSize({width:390,height:844});if(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('mobile overflow');
await b.close();if(errors.length)throw Error(errors.join('\n'));
const html=path.resolve('results/data-foundation/2111-blocks-ui-r1/df-v001/layer-workload-r1/index.html');const report={status:'PASS',checks,http:200,allEightVisibilityCombinations:true,continuousAcrossPP:true,stableColors:true,sections2and4Unchanged:true,mobileOverflow:false,errors,html_sha256:crypto.createHash('sha256').update(fs.readFileSync(html)).digest('hex')};fs.writeFileSync(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n');console.log(report);
})().catch(e=>{console.error(e);process.exit(1)});
