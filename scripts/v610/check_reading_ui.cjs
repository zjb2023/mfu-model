const fs=require('fs'),path=require('path'),{chromium}=require('playwright');
const root=path.resolve(__dirname,'../..'),service=JSON.parse(fs.readFileSync(root+'/results/v610/service.json'));
const origin=`http://127.0.0.1:${service.port}`,out=path.join(root,'results/v610/ui-reading-check');
fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'/home/zjb/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome',args:['--no-sandbox']});
 const checks=[],errors=[];let finished=false;function check(name,ok){checks.push({name,passed:!!ok});if(!ok)throw Error(name)}
 try{
  const p=await browser.newPage({viewport:{width:1440,height:1000},acceptDownloads:true});p.on('pageerror',e=>errors.push(e.message));
  await p.goto(origin+'/w37-report.html#task');
  check('eight historical-layout chapters',await p.locator('main > section').count()===8);
  check('current v610 title',(await p.title()).includes('v6.10'));
  check('methods comparison retained',(await p.locator('#why-dag').innerText()).includes('不是所有实现都简单三段相加'));
  check('no obsolete diagnostic amount',!(await p.locator('#why-dag').innerText()).includes('3.420'));
  for(const id of ['training','why-dag','task','calibration','development','results','diagnosis','limits']){
   await p.locator(`nav a[href="#${id}"]`).click();check('navigation:'+id,new URL(p.url()).pathname==='/w37-report.html'&&new URL(p.url()).hash==='#'+id);
  }
  await p.goto(origin+'/w37-report.html#calibration');
  check('brief uses current counts',(await p.locator('#calibration-brief').innerText()).includes('70192'));
  await p.locator('#calibration-evidence > summary').click();
  await p.locator('#fine-parameters').locator('..').locator('summary').first().click();
  check('6765 full source parameters',(await p.locator('#fine-parameters .table-status').innerText()).includes('6765'));
  await p.locator('#fine-parameters input').fill('forward');
  check('source parameter filter',(await p.locator('#fine-parameters tbody tr').count())>0);
  await p.locator('#fine-parameters input').fill('');
  await p.locator('#full-iter-choice').selectOption('100');
  check('current four-iteration graph',await p.locator('#full-iter-plots svg').count()>0);
  await p.goto(origin+'/w37-report.html#calibration');
  await p.evaluate(()=>document.getElementById('calibration').scrollIntoView({block:'start'}));
  await p.screenshot({path:out+'/report-calibration.png'});
  await p.locator('#parameter-atlas-entry a').click();
  check('standalone atlas',p.url().endsWith('/v610-parameter-atlas.html'));
  await p.waitForFunction(()=>typeof window.atlasState==='object');
  const meta=await p.evaluate(()=>({version:data.version,coverage:data.coverage,phases:Object.keys(data.phases).length}));
  check('atlas version and full coverage',meta.version==='v6.10'&&meta.coverage.full_bindings===70192&&meta.coverage.bound_keys===4387&&meta.phases===84);
  check('no historical version label',!(await p.locator('body').innerText()).includes('v685'));
  check('14 PP choices',await p.locator('#stage option').count()===14);
  await p.locator('#example-f13').click();
  check('interior PP example',await p.locator('#stage').inputValue()==='6');
  const info=await p.locator('#node-info').innerText();
  check('source value multiplier and provenance',info.includes('源细粒度计算')&&info.includes('继承形状倍率')&&info.includes('85 / 90'));
  check('no undefined numerical fields',!info.includes('NaN')&&!info.includes('undefined'));
  const state=await p.evaluate(()=>window.atlasState);
  check('selected nodes use real bindings',state.visible.some(n=>n.updated));
  await p.locator('#category').selectOption('all');await p.locator('#direction').selectOption('BWD');
  await p.locator('#mb').selectOption('2');await p.locator('#stage').selectOption('13');
  check('PP MB F/B selectors',(await p.locator('#phase-info').innerText()).includes('PP13 / rank208 / lane0 / MB2 / BWD'));
  await p.locator('[data-overview-entry]').click();check('entry node card',(await p.locator('#node-info').innerText()).includes('1265.388210'));
  await p.locator('[data-tail-summary="ag1"]').click();check('AG summary',(await p.locator('#node-info').innerText()).includes('AG1'));
  await p.screenshot({path:out+'/atlas.png'});
  await p.locator('header a').click();check('atlas returns current calibration',p.url().endsWith('/w37-report.html#calibration'));
  await p.goto(origin+'/w37-report.html#why-dag');await p.locator('#why-dag a[href="/v610-teaching-dag.html"]').first().click();
  await p.locator('#work-report-link').click();check('teaching returns current chapter',p.url().endsWith('/w37-report.html#why-dag'));
  const download= p.waitForEvent('download');await p.locator('#download-html').click();const file=await download;await file.saveAs(out+'/downloaded-report.html');
  await p.route('http://**/*',r=>r.abort());await p.route('https://**/*',r=>r.abort());
  await p.goto('file://'+out+'/downloaded-report.html#results');await p.locator('#full-iter-choice').selectOption('90');
  check('offline current graph',await p.locator('#full-iter-plots svg').count()>0);
  await p.goto('file://'+path.join(service.ui_root,'PARAMETER_ITER_ATLAS.html'));await p.locator('#example-b0').click();
  check('offline atlas bindings',(await p.locator('#node-info').innerText()).includes('源细粒度计算'));
  await p.setViewportSize({width:390,height:844});check('mobile no page-wide overflow',await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+2));
  check('no script errors',errors.length===0);
  finished=true;console.log('PASS',checks.length,'reading UI checks');
 }catch(error){errors.push(error.message);throw error}
 finally{fs.writeFileSync(out+'/acceptance.json',JSON.stringify({status:finished&&checks.every(c=>c.passed)&&errors.length===0?'PASS':'FAIL',checks,errors,origin,ui_root:service.ui_root},null,2));await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
