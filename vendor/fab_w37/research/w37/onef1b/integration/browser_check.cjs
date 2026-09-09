// Read-only UI acceptance against the running 8037 server and downloaded HTML.
const fs=require('fs');
const path=require('path');
const crypto=require('crypto');
const {pathToFileURL}=require('url');
const root=path.resolve(__dirname,'../../../..');
const out=path.join(root,'results/w37/A/integration-20260907');
const {chromium}=require(path.join(root,'results/w37/A/research-html-20260907/browser-check/node_modules/playwright'));
const origin='http://192.168.0.49:8037';
const expected={'physical-params':58,'compute-updates':144,'gradient-params':960,'version-iterations':20,'online-iterations':12,'scaleout-iterations':20};
const checks=[];
function check(name,ok,detail){checks.push({name,passed:!!ok,detail});if(!ok)throw Error(name+': '+JSON.stringify(detail));}
async function main(){
  fs.mkdirSync(path.join(out,'browser'),{recursive:true});
  const reportHash=()=>crypto.createHash('sha256').update(fs.readFileSync(path.join(out,'integrated_report.html'))).digest('hex');
  const initialHash=reportHash();
  const browser=await chromium.launch({headless:true,executablePath:'/home/zjb/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome',args:['--no-sandbox']});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1000}});
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    const response=await page.goto(origin+'/w37-report.html');
    check('HTTP_UTF8',response.status()===200&&response.headers()['content-type'].includes('charset=utf-8'));
    await page.locator('a[href="#binding-audit"]').first().click();
    check('binding_audit_anchor',new URL(page.url()).hash==='#binding-audit'&&await page.locator('#binding-audit').isVisible());
    check('binding_all_ranks_explained',(await page.locator('#binding-audit').innerText()).includes('224 rank、PP0–PP13、MB0–MB2'));
    await page.locator('a[href="#binding-results"]').first().click();
    check('binding_results_anchor',new URL(page.url()).hash==='#binding-results'&&await page.locator('#binding-results').isVisible());
    check('binding_candidate_metrics',(await page.locator('#binding-results').innerText()).includes('10.5616'));
    check('binding_regression_and_limit',(await page.locator('#binding-results').innerText()).includes('35.7889')&&(await page.locator('#binding-results').innerText()).includes('暂不升级正式版本'));
    await page.locator('#binding-results').screenshot({path:path.join(out,'browser/binding-results.png'),style:'.toc{visibility:hidden}'});
    check('node_map_before_both_parameter_tables',await page.evaluate(()=>{
      const figure=document.getElementById('parameter-node-map');
      return ['physical-params','compute-updates'].every(id=>!!(figure.compareDocumentPosition(document.getElementById(id))&Node.DOCUMENT_POSITION_FOLLOWING));
    }));
    check('node_map_six_identifier_fields',await page.locator('.node-map-fields article').count()===6);
    check('node_map_selected_backward_task',(await page.locator('.node-map-selected').innerText()).includes('B(MB0)'));
    await page.locator('#parameter-node-map').screenshot({path:path.join(out,'browser/parameter-node-map.png'),style:'.toc{visibility:hidden}'});
    check('calibration_diagram_visible',await page.locator('#v685-calibration-map svg').isVisible());
    check('calibration_diagram_names_and_costs',(await page.locator('#v685-calibration-map').textContent()).includes('dense_to_step_exit')&&(await page.locator('#v685-calibration-map').textContent()).includes('100.629 ms'));
    await page.locator('#v685-calibration-map svg').screenshot({path:path.join(out,'browser/calibration-map.png'),style:'.toc{visibility:hidden}'});
    check('parameter_system_visible',await page.locator('#parameter-system').isVisible());
    check('forward_version_difference_explained',(await page.locator('#parameter-system').innerText()).includes('0.3263 ms'));
    await page.locator('a[href="#pp-model-versions"]').click();
    check('communication_formula_anchor',new URL(page.url()).hash==='#pp-model-versions');
    await page.locator('a[href="#compute-binding-explained"]').click();
    check('compute_positions_anchor',new URL(page.url()).hash==='#compute-binding-explained');
    check('training_is_first_section',await page.locator('main > section').first().getAttribute('id')==='training');
    check('eight_technical_sections',await page.locator('main > section').count()===8);
    check('no_calendar_chapter_titles',!((await page.locator('main > section > h2').allTextContents()).join(' ').match(/周五|周末/)));
    await page.locator('a[href="#scaleout"]').first().click();
    check('two_scaleout_plots',await page.locator('#scaleout svg.scaleout-plot').count()===2);
    check('scaleout_numeric_labels',(await page.locator('#scaleout').innerText()).includes('84.8177%'));
    await page.locator('#why-dag').screenshot({path:path.join(out,'browser/why-dag.png')});
    check('three_complete_iteration_plots',await page.locator('#full-iter-plots svg').count()===3);
    check('each_view_explains_outer',await page.locator('.full-iter-accounting').count()===3);
    check('prediction_distinguishes_two_adjustments',(await page.locator('[data-view="prediction"] .full-iter-accounting').innerText()).includes('604.440 ms')&&(await page.locator('[data-view="prediction"] .full-iter-accounting').innerText()).includes('1372.453 ms'));
    check('no_tail_hatch_implies_adjustment_location',await page.locator('#full-iter-plots pattern').count()===0);
    const fixedPrediction=await page.locator('[data-view="prediction"] svg').evaluate(e=>e.outerHTML);
    for(const iteration of ['85','90','95','100']){
      await page.selectOption('#full-iter-choice',iteration);
      check('full_iter_selected:'+iteration,(await page.locator('#full-iter-score').innerText()).includes('第 '+iteration+' 轮'));
      check('prediction_fixed:'+iteration,(await page.locator('[data-view="prediction"] svg').evaluate(e=>e.outerHTML))===fixedPrediction);
      for(const [view,count]of [['prediction',84],['target',84],['source',128]])check('FB_grid:'+iteration+':'+view,await page.locator(`[data-view="${view}"] rect[data-kind="forward"], [data-view="${view}"] rect[data-kind="backward"]`).count()===count);
      check('full_iter_ledger_three_rows:'+iteration,await page.locator('#full-iter-ledger tbody tr').count()===3);
    }
    await page.selectOption('#full-iter-choice','85');
    await page.locator('[data-view="target"] rect[data-kind="forward"]').first().click();
    check('timeline_event_selection',(await page.locator('#full-iter-selection').innerText()).includes('全部16rank'));
    await page.locator('#full-iter-comms').uncheck();
    check('hide_communication_and_optimizer',await page.locator('#full-iter-plots rect[data-kind="dp"], #full-iter-plots rect[data-kind="optimizer"]').count()===0);
    await page.locator('#full-iter-comms').check();
    check('restore_predicted_optimizer',await page.locator('[data-view="prediction"] rect[data-kind="optimizer"]').count()===42);
    check('trace_does_not_invent_optimizer',await page.locator('[data-view="target"] rect[data-kind="optimizer"], [data-view="source"] rect[data-kind="optimizer"]').count()===0);
    for(const window of ['fb','tail','all']){await page.selectOption('#full-iter-window',window);check('range_selection:'+window,await page.locator('#full-iter-plots svg').count()===3);}
    await page.locator('[data-view="prediction"]').screenshot({path:path.join(out,'browser/full-iter-prediction.png')});
    await page.locator('[data-view="target"]').screenshot({path:path.join(out,'browser/full-iter-target.png')});
    await page.locator('[data-view="source"]').screenshot({path:path.join(out,'browser/full-iter-source.png')});
    await page.locator('#full-iter > h3').scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(out,'browser/full-iter-overview.png')});
    await page.evaluate(()=>scrollTo({top:0,behavior:'instant'}));
    await page.screenshot({path:path.join(out,'browser/desktop.png')});
    const anchors=await page.locator('.toc a').evaluateAll(xs=>xs.map(x=>x.getAttribute('href')));
    for(const href of anchors){await page.locator('.toc a[href="'+href+'"]').click();check('navigation:'+href,await page.evaluate(id=>!!document.querySelector(id),href));}
    await page.locator('#expand-content').click();
    check('expand_all',await page.locator('details:not([open])').count()===0);
    await page.evaluate(async()=>{await Promise.all([...document.images].map(img=>{img.loading='eager';return img.decode()}))});
    check('all_embedded_images_decoded',await page.locator('img').evaluateAll(xs=>xs.length===5&&xs.every(x=>x.complete&&x.naturalWidth>0)));
    for(const [id,count] of Object.entries(expected)){
      const box=page.locator('#'+id);
      check('table_count:'+id,(await box.locator('.table-status').innerText()).includes('共 '+count+' 行'));
      const next=box.locator('[data-next]');
      if(count>20){await next.click();check('next_page:'+id,(await box.locator('.table-status').innerText()).includes('第 2 /'));await box.locator('[data-prev]').click();check('previous_page:'+id,(await box.locator('.table-status').innerText()).includes('第 1 /'));}
      const search=box.locator('input');await search.fill('不存在的筛选结果');
      check('empty_search:'+id,await box.locator('tbody tr').count()===0);
      await search.fill('');check('restore_search:'+id,await box.locator('tbody tr').count()===Math.min(20,count));
    }
    for(const scope of ['primary_causal_walk_forward','all_including_initialization']){
      await page.selectOption('#result-scope',scope);
      check('scope_visibility:'+scope,await page.locator('[data-result-scope="'+scope+'"]').isVisible());
      check('scope_only_one',await page.locator('[data-result-scope]:visible').count()===1);
    }
    await page.locator('#graph').screenshot({path:path.join(out,'browser/graph.png')});
    await page.locator('#calibration h2').scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(out,'browser/calibration.png')});
    const links=[...new Set(await page.locator('a[href]').evaluateAll(xs=>xs.map(x=>x.href)))];
    const urls=[...new Set(links.filter(x=>x.startsWith(origin)).map(x=>x.split('#')[0]))];
    const linkResults=[];
    for(const url of urls){const r=await page.request.get(url);linkResults.push({url,status:r.status()});check('link:'+url,r.ok());}
    // Check off-page fragment targets, not only the HTTP response.
    const probe=await browser.newPage();
    for(const url of links.filter(x=>x.startsWith(origin)&&x.includes('#')&&!x.startsWith(origin+'/w37-report.html#'))){await probe.goto(url);check('external_fragment:'+url,await probe.evaluate(()=>!!document.getElementById(decodeURIComponent(location.hash.slice(1)))));}
    await probe.close();
    // Legacy URLs must reveal content that is now grouped under a technical chapter.
    for(const [anchor,chapter] of [['baseline','results'],['summary','results'],['weekend','development'],['scaleout','limits'],['tool-todo','limits'],['sources','limits'],['graph','task'],['three-case-parameters','calibration']]){
      await page.locator('#collapse-content').click();
      await page.evaluate(id=>{location.hash=id},anchor);
      await page.waitForFunction(id=>{const e=document.getElementById(id);return e&&e.getClientRects().length>0},anchor);
      check('legacy_anchor_reveals:'+anchor,await page.locator('#'+anchor).isVisible());
      check('legacy_anchor_chapter:'+anchor,await page.locator('#'+anchor).evaluate(e=>e.closest('section').id)===chapter);
    }
    await page.goto(origin+'/w37-report.html#baseline');
    check('direct_old_baseline_url',(await page.locator('#baseline').innerText()).includes('75.8')&&await page.locator('#baseline').isVisible());
    await page.goto(origin+'/w37-report.html');
    await page.locator('#collapse-content').click();
    check('collapse_all',await page.locator('details[open]').count()===0);
    const downloadPromise=page.waitForEvent('download');await page.locator('#download-html').click();
    const download=await downloadPromise;check('technical_download_name',download.suggestedFilename()==='W37_分布式训练与MFU性能预测.html');const downloaded=path.join(out,'browser/downloaded-report.html');await download.saveAs(downloaded);
    check('download_is_complete',fs.statSync(downloaded).size>1000000);
    const offline=await browser.newContext({offline:true,viewport:{width:1440,height:1000}});
    const local=await offline.newPage();const offlineErrors=[];local.on('pageerror',e=>offlineErrors.push(e.message));
    await local.goto(pathToFileURL(downloaded).href);await local.locator('#expand-content').click();
    await local.evaluate(async()=>{await Promise.all([...document.images].map(img=>{img.loading='eager';return img.decode()}))});
    check('offline_eight_chapters',await local.locator('main > section').count()===8);
    check('offline_binding_results',(await local.locator('#binding-results').innerText()).includes('10.5616'));
    check('offline_images',await local.locator('img').evaluateAll(xs=>xs.length===5&&xs.every(x=>x.complete&&x.naturalWidth>0)));
    await local.locator('#gradient-params input').fill('90');
    check('offline_table_operates',await local.locator('#gradient-params tbody tr').count()>0);
    await local.selectOption('#result-scope','primary_causal_walk_forward');
    check('offline_scope_operates',await local.locator('[data-result-scope="primary_causal_walk_forward"]').isVisible());
    await local.selectOption('#full-iter-choice','100');
    check('offline_full_iter_switch',(await local.locator('#full-iter-score').innerText()).includes('第 100 轮'));
    check('offline_full_iter_three_charts',await local.locator('#full-iter-plots svg').count()===3);
    check('offline_no_JS_error',offlineErrors.length===0,offlineErrors);await offline.close();
    await page.setViewportSize({width:390,height:844});await page.goto(origin+'/w37-report.html');
    check('mobile_collapsed_no_page_overflow',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    await page.screenshot({path:path.join(out,'browser/mobile.png')});
    await page.locator('#expand-content').click();
    check('mobile_expanded_no_page_overflow',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    check('browser_no_JS_errors',errors.length===0,errors);
    await page.goto(origin+'/research.html');
    await page.locator('a[href="/w37-report.html"]').click();check('research_home_entry',page.url()===origin+'/w37-report.html');
    check('report_unchanged_during_browser_checks',reportHash()===initialHash);
    fs.writeFileSync(path.join(out,'browser_validation.json'),JSON.stringify({status:'PASS',html_sha256:initialHash,checks,check_count:checks.length,links:linkResults,desktop:[1440,1000],mobile:[390,844],offline_download_tested:true},null,2)+'\n');
    console.log(JSON.stringify({status:'PASS',checks:checks.length,unique_HTTP_links:urls.length}));
  }finally{await browser.close()}
}
main().catch(e=>{fs.writeFileSync(path.join(out,'browser_validation.json'),JSON.stringify({status:'FAIL',checks,error:String(e)},null,2)+'\n');console.error(e);process.exitCode=1;});
