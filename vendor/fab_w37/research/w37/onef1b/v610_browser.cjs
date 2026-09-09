// Browser acceptance for the current release; never mutates sealed artifacts.
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {pathToFileURL}=require('url');
const root=path.resolve(__dirname,'../../..'),out=path.join(root,'results/w37/A/v610-webcheck');
const {chromium}=require(path.join(root,'results/w37/A/research-html-20260907/browser-check/node_modules/playwright'));
const origin='http://192.168.0.49:8037',checks=[];
const hash=()=>crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'results/w37/A/v610-release/render/index.html'))).digest('hex');
function check(name,value){checks.push({name,passed:!!value});if(!value)throw Error(name);}
async function main(){
 fs.mkdirSync(out,{recursive:true});const initial=hash();
 const browser=await chromium.launch({headless:true,executablePath:'/home/zjb/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome',args:['--no-sandbox']});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(origin+'/research.html');check('current_home_version',(await page.locator('h1').innerText()).includes('v6.10'));
  await page.locator('a[href="/w37-report.html"]').click();check('home_opens_current',page.url()===origin+'/w37-report.html');
  check('current_title',(await page.locator('h1').innerText()).includes('v6.10'));
  check('current_metric_cards',(await page.locator('header').innerText()).includes('10.5616%'));
  check('seven_chapters',await page.locator('main > section').count()===7);
  await page.screenshot({path:path.join(out,'desktop.png')});
  for(const id of ['training','task','full-iter','calibration','results','limits','sources']){
   await page.locator('nav a[href="#'+id+'"]').click();check('nav:'+id,new URL(page.url()).hash==='#'+id&&await page.locator('#'+id).isVisible());
  }
  for(const id of ['binding-audit','binding-results']){await page.goto(origin+'/w37-report.html#'+id);check('recent_anchor:'+id,await page.locator('#'+id).evaluate(e=>!!e));}
  check('interior_PP_example',(await page.locator('#parameter-node-map').innerText()).includes('r96:s6:l0:fwd0'));
  check('parameter_cards_readable',await page.locator('.node-map-transfer article').evaluateAll(xs=>xs.length===3&&xs.every(x=>x.getBoundingClientRect().width>250&&x.getBoundingClientRect().height<500)));
  check('interior_layer_example',(await page.locator('#parameter-node-map').innerText()).includes('22'));
  await page.locator('#parameter-node-map').screenshot({path:path.join(out,'node-parameter.png'),style:'.toc{visibility:hidden}'});
  await page.locator('#expand-content').click();check('expand_all',await page.locator('details:not([open])').count()===0);
  check('fine_6765',(await page.locator('#fine-parameters .table-status').innerText()).includes('6765'));
  const first=await page.locator('#fine-parameters tbody').innerText();await page.locator('#fine-parameters [data-next]').click();
  check('next_parameter_page',first!==await page.locator('#fine-parameters tbody').innerText());await page.locator('#fine-parameters [data-prev]').click();check('previous_parameter_page',first===await page.locator('#fine-parameters tbody').innerText());
  await page.locator('#fine-parameters input').fill('not-a-real-parameter');check('empty_parameter_filter',await page.locator('#fine-parameters tbody tr').count()===0);
  await page.locator('#fine-parameters input').fill('forward');check('parameter_filter',await page.locator('#fine-parameters tbody tr').count()>0);await page.locator('#fine-parameters input').fill('');
  const fixed=await page.locator('[data-view="prediction"] svg').evaluate(e=>e.outerHTML);
  for(const it of ['85','90','95','100']){
   await page.selectOption('#full-iter-choice',it);check('iteration:'+it,(await page.locator('#full-iter-score').innerText()).includes('第 '+it+' 轮'));
   check('prediction_fixed:'+it,(await page.locator('[data-view="prediction"] svg').evaluate(e=>e.outerHTML))===fixed);
   for(const [view,n] of [['prediction',84],['target',84],['source',128]])check('FB:'+it+':'+view,await page.locator('[data-view="'+view+'"] rect[data-kind="forward"], [data-view="'+view+'"] rect[data-kind="backward"]').count()===n);
  }
  check('new_prediction_time',(await page.locator('[data-view="prediction"]').innerText()).includes('22.930'));
  check('new_prediction_label',(await page.locator('[data-view="prediction"] h3').innerText()).includes('v6.10'));
  await page.selectOption('#full-iter-choice','85');check('new_iteration_error',(await page.locator('#full-iter-score').innerText()).includes('9.861'));
  for(const window of ['fb','tail','all']){await page.selectOption('#full-iter-window',window);check('window:'+window,await page.locator('#full-iter-plots svg').count()===3);}
  await page.locator('[data-view="prediction"] rect[data-kind="forward"]').first().click();check('timeline_selection',(await page.locator('#full-iter-selection').innerText()).includes('全部16rank'));
  await page.locator('#full-iter-comms').uncheck();check('hide_comms',await page.locator('#full-iter-plots rect[data-kind="optimizer"]').count()===0);
  await page.locator('#full-iter-comms').check();check('restore_opt',await page.locator('[data-view="prediction"] rect[data-kind="optimizer"]').count()===42);
  check('no_invented_trace_opt',await page.locator('[data-view="target"] rect[data-kind="optimizer"]').count()===0);
  await page.locator('#results').screenshot({path:path.join(out,'results.png'),style:'.toc{visibility:hidden}'});
  const urls=[...new Set(await page.locator('a[href]').evaluateAll(xs=>xs.map(a=>a.href).filter(x=>x.startsWith('http')).map(x=>x.split('#')[0])))];
  const links=[];for(const url of urls){const r=await page.request.get(url);check('HTTP:'+url,r.ok());links.push({url,status:r.status()});}
  for(const anchor of ['scaleout','tool-todo','baseline']){await page.goto(origin+'/w37-report.html#'+anchor);await page.waitForURL('**/w37-report-history.html#'+anchor);check('legacy_history:'+anchor,await page.locator('#'+anchor).isVisible());}
  await page.goto(origin+'/v610.html');check('version_alias',(await page.locator('h1').innerText()).includes('v6.10'));
  await page.locator('#collapse-content').click();check('collapse_all',await page.locator('details[open]').count()===0);
  const promise=page.waitForEvent('download');await page.locator('#download-html').click();const download=await promise;check('download_name',download.suggestedFilename()==='v6.10_计算图MFU预测模型.html');
  const local=path.join(out,'download.html');await download.saveAs(local);
  const offline=await browser.newContext({offline:true,viewport:{width:1440,height:1000}});const file=await offline.newPage();const offlineErrors=[];file.on('pageerror',e=>offlineErrors.push(e.message));
  await file.goto(pathToFileURL(local).href);check('offline_current_title',(await file.locator('h1').innerText()).includes('v6.10'));
  await file.selectOption('#full-iter-choice','100');check('offline_iter',(await file.locator('#full-iter-score').innerText()).includes('第 100 轮'));
  await file.locator('#expand-content').click();await file.locator('#fine-parameters input').fill('backward');check('offline_parameters',await file.locator('#fine-parameters tbody tr').count()>0);check('offline_no_errors',offlineErrors.length===0);await offline.close();
  await page.setViewportSize({width:390,height:844});await page.goto(origin+'/w37-report.html');check('mobile_no_overflow',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  await page.screenshot({path:path.join(out,'mobile.png')});await page.locator('#expand-content').click();check('mobile_expanded_no_overflow',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  check('no_JS_errors',errors.length===0);check('immutable_during_checks',hash()===initial);
  fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify({status:'PASS',version:'v6.10',html_sha256:initial,checks,check_count:checks.length,links,offline:true,mobile:true},null,2)+'\n');console.log(JSON.stringify({status:'PASS',checks:checks.length,links:links.length}));
 }finally{await browser.close()}
}
main().catch(e=>{fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify({status:'FAIL',checks,error:String(e)},null,2)+'\n');console.error(e);process.exitCode=1});
