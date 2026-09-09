const fs=require('fs'),path=require('path');
const root=path.resolve(__dirname,'../../../..');
const {chromium}=require(path.join(root,'results/w37/A/research-html-20260907/browser-check/node_modules/playwright'));
const out=path.join(root,'results/w37/A/history-scaleout-20260908'),checks=[];
function check(name,passed){checks.push({name,passed:!!passed});if(!passed)throw Error(name);}
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'/home/zjb/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome',args:['--no-sandbox']});
 try{
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 // No live 8038 dependency is permitted while reading the imported report.
 await page.route('http://192.168.0.49:8038/**',r=>r.abort());
 const r=await page.goto('http://192.168.0.49:8037/w37-report-history.html#scaleout');check('history HTTP200 UTF8',r.status()===200&&r.headers()['content-type'].includes('utf-8'));
 check('six summary methods',await page.locator('#scaleout > .scroll tbody tr').count()===6);
 check('T3 and M3 imported',(await page.locator('#scaleout').innerText()).includes('T3')&&(await page.locator('#scaleout').innerText()).includes('M3'));
 check('old explorer replaced',await page.locator('#scaleout-iterations').count()===0);
 const frame=page.frameLocator('#scaleout-report-frame');await frame.locator('#method-choice').waitFor();
 check('local embedded report',await frame.locator('h1').count()===1);
 const methods=await frame.locator('#method-choice option').evaluateAll(es=>es.map(e=>e.value));check('six methods available',methods.length===6);
 for(const method of methods){await frame.locator('#method-choice').selectOption(method);check('method '+method,(await frame.locator('#result-chart').innerHTML()).length>100);}
 const metrics=await frame.locator('#metric-choice option').evaluateAll(es=>es.map(e=>e.value));
 for(const metric of metrics){await frame.locator('#metric-choice').selectOption(metric);check('metric '+metric,(await frame.locator('#result-chart').innerHTML()).length>100);}
 for(const pp of ['2','16'])for(const mb of ['1','4']){await frame.locator('#pp-choice').selectOption(pp);await frame.locator('#mb-choice').selectOption(mb);check('schedule '+pp+'x'+mb,(await frame.locator('#pipeline-chart').innerHTML()).length>100);}
 const nav=await frame.locator('nav a[href^="#"]').evaluateAll(es=>es.map(e=>e.getAttribute('href')));
 for(const href of nav){await frame.locator('nav a[href="'+href+'"]').click();check('child nav '+href,await frame.locator(href).count()===1);}
 const download=page.waitForEvent('download');await frame.locator('#download').click();const d=await download;await d.saveAs(path.join(out,'download168.csv'));check('download 168 rows',fs.readFileSync(path.join(out,'download168.csv'),'utf8').trim().split('\n').length===169);
 const urls=await frame.locator('a[href*="/evidence/"]').evaluateAll(es=>es.map(e=>e.href));check('eleven local evidence links',urls.length===11);
 for(const u of urls){const res=await page.request.get(u);check('evidence '+new URL(u).pathname,res.status()===200);}
 await page.goto('http://192.168.0.49:8037/w37-report-history.html#calibration');check('other historical tables intact',await page.locator('#physical-params tbody tr').count()>0);
 await page.goto('http://192.168.0.49:8037/w37-report-history.html#limits');check('limits accessible',await page.locator('#limits').count()===1);
 await page.goto('http://192.168.0.49:8037/w37-report-history.html#scaleout');await page.screenshot({path:path.join(out,'desktop.png')});
 await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(out,'mobile.png')});check('parent mobile width',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
 const current=await page.request.get('http://192.168.0.49:8037/w37-report.html');check('current report retained',(await current.text()).includes('v6.10'));
 await page.goto('http://192.168.0.49:8037/w37-report-history.html#why-dag');
 await page.locator('#four-gpu-demo-index a').first().click();
 check('work report opens standalone demo',page.url().endsWith('/DAG_4GPU_DEMO.html'));
 const demo=page;
 check('historical time error clarified',(await demo.locator('#historical-result').innerText()).includes('16.823672%')&&(await demo.locator('#historical-result').innerText()).includes('不是 MFU 利用率'));
 await demo.locator('#work-report-link').click();
 check('demo returns to work report modeling chapter',page.url().endsWith('/w37-report-history.html#why-dag'));
 await page.goto('http://192.168.0.49:8037/w37-report-history.html#observed-mfu');
 check('MFU table has four measured iterations',await page.locator('#observed-mfu-table tbody tr').count()===4);
 check('peak and measured clock distinction',(await page.locator('#observed-mfu').innerText()).includes('112 PFLOP/s')&&(await page.locator('#observed-mfu').innerText()).includes('不是 trace 直接测得'));
 for(const href of await page.locator('#observed-mfu a').evaluateAll(es=>es.map(e=>e.href))){const res=await page.request.get(href);check('MFU evidence link '+new URL(href).pathname,res.status()===200);}
 const account=await (await page.request.get('http://192.168.0.49:8037/results/w37/A/history-scaleout-20260908/observed_mfu_accounting.json')).json();
 check('all measured MFU and compute formulas',account.rows.every(r=>Math.abs(r.training_mfu_pct-r.training_effective_pflops/112*100)<1e-10&&Math.abs(r.training_per_gpu_tflops*224/1000-r.training_effective_pflops)<1e-10&&Math.abs(r.training_effective_pflops*r.actual_training_seconds-account.model_flops_per_iteration/1e15)<1e-10&&r.profiler_interval_ratio_pct>r.training_mfu_pct));
 await page.screenshot({path:path.join(out,'observed-mfu-mobile.png')});
 check('MFU mobile page width',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
 await page.setViewportSize({width:1440,height:1100});
 await page.locator('#observed-mfu').screenshot({path:path.join(out,'observed-mfu-desktop.png')});
 await page.goto('http://192.168.0.49:8037/w37-report-history.html#limits');
 check('MLPerf TODO inside limits',await page.locator('#limits #mlperf-deepseek-todo tbody tr').count()===9);
 check('four recipes and missing R10',await page.locator('#mlperf-recipe-table tbody tr').count()===4&&(await page.locator('#mlperf-recipe-table tbody tr').first().innerText()).includes('未匹配'));
 check('MFU estimates imported',(await page.locator('#mlperf-recipe-table').innerText()).includes('32.48%')&&(await page.locator('#mlperf-recipe-table').innerText()).includes('11.97%'));
 for(const href of await page.locator('#mlperf-deepseek-todo a').evaluateAll(es=>es.map(e=>e.href))){const res=await page.request.get(href);check('recipe source '+new URL(href).pathname,res.status()===200);}
 await page.locator('#mlperf-deepseek-todo a').last().click();
 check('MLPerf task evidence link readable',(await page.locator('body').innerText()).includes('唯一下一步'));
 await page.goto('http://192.168.0.49:8037/w37-report-history.html#calibration');await page.reload();
 check('chapter04 detail collapsed by default',!(await page.locator('#calibration-evidence').evaluate(e=>e.open)));
 check('chapter04 concise three changes',await page.locator('#calibration-brief > ul > li').count()===3);
 for(const hash of ['parameter-system','physical-params','pp-model-versions']){await page.locator('#calibration-brief a[href="#'+hash+'"]').click();check('legacy evidence anchor opens '+hash,await page.locator('#calibration-evidence').evaluate(e=>e.open));}
 await page.goto('http://192.168.0.49:8037/w37-report-history.html#calibration');await page.reload();await page.locator('#calibration-evidence > summary').click();check('manual evidence expansion',await page.locator('#calibration-evidence').evaluate(e=>e.open));await page.locator('#calibration-evidence > summary').click();check('manual evidence collapse',!(await page.locator('#calibration-evidence').evaluate(e=>e.open)));
 await page.goto('http://192.168.0.49:8037/w37-report-history.html#v685-time-mfu');
 await page.locator('#v685-time-mfu img').scrollIntoViewIfNeeded();await page.locator('#v685-time-mfu img').evaluate(e=>e.decode());
 check('time MFU figure loaded',await page.locator('#v685-time-mfu img').evaluate(e=>e.naturalWidth>0));
 const fig=await (await page.request.get('http://192.168.0.49:8037/results/w37/A/history-scaleout-20260908/v685_time_and_mfu.json')).json();
 check('same four iteration MFU values',fig.iterations.join(',')==='85,90,95,100'&&Math.abs(fig.predicted_mfu_pct_mean-3.3098615149285155)<1e-9&&Math.abs(fig.actual_mfu_pct_mean-2.955109205598223)<1e-9);
 const svgResponse=await page.request.get('http://192.168.0.49:8037/results/w37/A/history-scaleout-20260908/v685_time_and_mfu.svg');
 check('SVG includes time and both MFUs',svgResponse.status()===200&&(await svgResponse.text()).includes('3.309862%')&&(await svgResponse.text()).includes('2.955109%')&&(await svgResponse.text()).includes('25.491 s'));
 await page.setViewportSize({width:1440,height:1100});await page.locator('#v685-time-mfu').screenshot({path:path.join(out,'v685-time-mfu.png')});
 check('no JavaScript errors',errors.length===0);
 }finally{await browser.close();fs.writeFileSync(path.join(out,'browser_acceptance.json'),JSON.stringify({passed:checks.every(c=>c.passed),checks},null,2));}
 console.log(checks.length+' browser checks PASS');
})().catch(e=>{console.error(e);process.exit(1)});
