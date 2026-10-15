const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {chromium}=require('/home/zjb/Desktop/worktrees/mfu-w37-v610/node_modules/playwright');
(async()=>{
 const root=path.resolve(__dirname,'../..'),out=path.join(root,'results/data-foundation/mccl-fb-structure-r1');
 const browser=await chromium.launch({headless:true,executablePath:'/home/zjb/.cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell',args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[],rows=[];
  page.on('pageerror',e=>errors.push(e.message));
  const response=await page.goto('http://192.168.8.16:43312/df-v001/mccl-fb-structure-r1/',{waitUntil:'load'});
  if(!response.ok())throw Error('HTTP failed');
  for(let it=155;it<=200;it+=5)for(let u=0;u<4;u++)for(const ph of ['F','B']){
   await page.selectOption('#iter',String(it));await page.selectOption('#unit',String(u));
   await page.click(ph==='F'?'#fbtn':'#bbtn');let last='',axis=null,positions=null;
   const keys=ph==='F'?['all','forward_cp','forward_ep']:['all','recompute_cp','recompute_ep','backward_ep','backward_cp'];
   const menu=await page.locator('#region option').evaluateAll(es=>es.map(e=>e.value));
   if(JSON.stringify(menu)!==JSON.stringify(keys))throw Error('Menu order');
   for(const key of keys){
    await page.selectOption('#region',key);
    const actual=await page.locator('#timeline').innerHTML();
    if(last===actual)throw Error('Unchanged region '+it+'/'+u+'/'+ph+'/'+key);
    last=actual;
    const count=await page.locator('#timeline rect.selected').count();
    const expected=await page.evaluate(({it,u,ph,key})=>DATA[it][ph][u].reduce((n,r)=>n+(key==='all'?new Set(Object.values(r.regions).flat().map(e=>e.id)).size:r.regions[key].length),0),{it,u,ph,key});
    if(count!==expected||/NaN|undefined/.test(actual))throw Error('Bad events '+key);
    const rankLabels=await page.locator('#timeline text').allTextContents();
    const coords=await page.locator('#timeline rect').evaluateAll(es=>es.map(e=>[e.getAttribute('x'),e.getAttribute('width')]));
    if(axis!==null&&(JSON.stringify(rankLabels)!==axis||JSON.stringify(coords)!==positions))throw Error('Time axis shifted on interval switch');
    axis=JSON.stringify(rankLabels);positions=JSON.stringify(coords);
    for(let r=8;r<16;r++)if(!rankLabels.includes('rank '+r))throw Error('Missing rank '+r);
    rows.push({it,u,ph,key,events:count});
   }
  }
  if(errors.length)throw Error(errors.join('\n'));
  fs.writeFileSync(path.join(out,'browser-verification.json'),JSON.stringify({status:'PASS',tested_utc:new Date().toISOString(),browser:'Chromium headless',combinations:rows.length,page_errors:errors,checks:['chronological menu','fixed layer axis and event positions','real select change updates SVG','selected event counts','all eight ranks','no NaN','no page errors'],page_sha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(out,'index.html'))).digest('hex'),rows},null,2));
  console.log('PASS browser F/B interval switching',rows.length,'combinations');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
