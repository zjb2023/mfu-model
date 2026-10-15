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
  await page.click('#version-old');
  if(await page.locator('#new-view').isVisible())throw Error('New view still visible in old mode');
  const legacyChecks=[];
  for(const ph of ['F','B']){
   const suffix=ph==='F'?'cp-ep8-structure-r1/':'b-four-layers-r10/';
   await page.click(ph==='F'?'#legacy-f':'#legacy-b');
   await page.waitForFunction(s=>document.querySelector('#legacy-frame').contentDocument?.URL.endsWith(s),suffix);
   const frame=await page.locator('#legacy-frame').contentFrame();
   await frame.locator('body').waitFor({state:'visible'});
   const text=await frame.locator('body').innerText();if(text.length<100)throw Error('Empty legacy page');
   legacyChecks.push({phase:ph,url:suffix,body_text_length:text.length});
   if(await page.locator('#iter').inputValue()!=='60')throw Error('Wrong legacy iteration');
   for(let u=0;u<4;u++){
    await page.selectOption('#unit',String(u));let axis=null,coords=null;
    for(const key of ph==='F'?['all','forward_cp','forward_ep']:['all','recompute_cp','recompute_ep','backward_ep','backward_cp']){
     await page.selectOption('#region',key);
     const current=await page.locator('#timeline rect').evaluateAll(es=>es.map(e=>[e.getAttribute('x'),e.getAttribute('width')]));
     const labels=await page.locator('#timeline text').allTextContents();
     if(axis!==null&&(axis!==JSON.stringify(labels)||coords!==JSON.stringify(current)))throw Error('Old time axis moved');
     axis=JSON.stringify(labels);coords=JSON.stringify(current);
     const count=await page.locator('#timeline rect.selected').count();
     const expected=await page.evaluate(({ph,u,key})=>DATA['60'][ph][u].reduce((n,r)=>n+(key==='all'?new Set(Object.values(r.regions).flat().map(e=>e.id)).size:r.regions[key].length),0),{ph,u,key});
     if(count!==expected||!count)throw Error('Legacy events missing');
     for(let rank=8;rank<16;rank++)if(!labels.includes('rank '+rank))throw Error('Legacy rank missing');
     rows.push({version:'old',it:60,u,ph,key,events:count});
    }
   }
  }
  await page.click('#version-new');
  if(!await page.locator('#new-view').isVisible()||await page.locator('#legacy-view').isVisible())throw Error('Version return failed');
  await page.selectOption('#region','backward_ep');
  if(!await page.locator('#timeline rect.selected').count())throw Error('Timeline broken after old view');
  if(errors.length)throw Error(errors.join('\n'));
  fs.writeFileSync(path.join(out,'browser-verification.json'),JSON.stringify({status:'PASS',tested_utc:new Date().toISOString(),browser:'Chromium headless',combinations:rows.length,legacyChecks,page_errors:errors,checks:['old/new version switching','legacy F/B iframe loaded','return to new B timeline','chronological menu','fixed layer axis and event positions','real select change updates SVG','selected event counts','all eight ranks','no NaN','no page errors'],page_sha256:crypto.createHash('sha256').update(fs.readFileSync(path.join(out,'index.html'))).digest('hex'),rows},null,2));
  console.log('PASS browser F/B interval switching',rows.length,'combinations');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
