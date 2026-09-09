const fs=require('fs'),path=require('path');
const root=path.resolve(__dirname,'../../../..');
const {chromium}=require(path.join(root,'results/w37/A/research-html-20260907/browser-check/node_modules/playwright'));
const out=path.join(root,'results/w37/A/extrapolation-plan'),checks=[];
function check(name,ok){checks.push({name,passed:!!ok});if(!ok)throw Error(name);}
(async()=>{
const browser=await chromium.launch({headless:true,executablePath:'/home/zjb/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome',args:['--no-sandbox']});
try{
const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
page.on('pageerror',e=>errors.push(e.message));
const url='http://192.168.0.49:8037/docs/w37/1f1b/EXTRAPOLATION_EXPERIMENT_PLAN.html';
const response=await page.goto(url);check('HTTP 200 UTF-8',response.status()===200&&response.headers()['content-type'].includes('utf-8'));
check('Chinese title',(await page.locator('header h1').innerText()).includes('MFU 外推实验方案'));
check('16 GPU fourteen configurations',await page.locator('#sixteen-doc table').first().locator('tbody tr').count()===14);
check('16 GPU six core',await page.locator('#sixteen-doc table').first().locator('tbody tr').filter({hasText:'核心'}).count()===6);
check('removed diagrams and controls',await page.locator('#sixteen-charts svg, #s16-controls, #s16-comparison').count()===0);
check('fourteen telemetry rows',await page.locator('#s16-telemetry tbody tr').count()===14);
check('DeepEP off all groups',await page.locator('#s16-telemetry tbody tr td:nth-child(7)').filter({hasText:'不采'}).count()===14);
check('MTLINK off all groups',await page.locator('#s16-telemetry tbody tr td:nth-child(8)').filter({hasText:'不采'}).count()===14);
check('profiler required all groups',await page.locator('#s16-telemetry tbody tr td:nth-child(5)').filter({hasText:'必采'}).count()===14);
const candidates=await page.locator('#sixteen-doc table').first().locator('tbody tr td:first-child').allTextContents();
const collection=await page.locator('#s16-telemetry tbody tr th').allTextContents();check('matrix and candidate IDs match',JSON.stringify(candidates)===JSON.stringify(collection));
for(const id of ['P16-PP1','P16-CP2','P16-TP2','P16-EP4'])check('merged '+id,candidates.includes(id));
await page.locator('#data-sources summary').click();check('provenance visible',(await page.locator('#data-sources').innerText()).includes('不是当前 v610 计算参数的来源'));await page.locator('#data-sources summary').click();
check('CP2 boundary',(await page.locator('#sixteen').innerText()).includes('需要 32 卡'));
check('recommended 11',await page.locator('#rows tr').count()===11);
await page.locator('#all').click();check('all 18',await page.locator('#rows tr').count()===18);
const cards=await page.locator('#rows tr td:nth-child(2)').allTextContents();const sizes=cards.map(x=>+x.split('/')[0]);check('no small GPU rows',sizes.every(n=>n>=32));check('GPU ascending',sizes.every((n,i)=>!i||n>=sizes[i-1]));
for(const [role,n]of [['calibration_candidate',12],['development_regression',2],['heldout_candidate',4]]){await page.selectOption('#role',role);check('filter '+role,await page.locator('#rows tr').count()===n);}
await page.selectOption('#role','');await page.locator('#search').fill('B32-EP4');check('search',await page.locator('#rows tr').count()===1);
await page.locator('#search').fill('no-match-xyz');check('empty state',(await page.locator('#count').innerText()).includes('没有匹配'));
await page.locator('#reset').click();check('reset',await page.locator('#rows tr').count()===11);
await page.locator('#all').click();await page.locator('#recommended').click();check('recommended button',await page.locator('#rows tr').count()===11);
for(const id of ['sixteen','roadmap','matrix','telemetry','memory','fulltext']){await page.locator(`nav a[href="#${id}"]`).click();check('nav '+id,new URL(page.url()).hash==='#'+id);}
await page.locator('#full-details > summary').click();check('complete text',(await page.locator('#full-details').innerText()).includes('B32-TP2'));
const anchors=await page.locator('a[href^="#"]').evaluateAll(as=>as.map(a=>a.getAttribute('href').slice(1)));
for(const id of anchors)check('anchor '+id,await page.evaluate(id=>!!document.getElementById(decodeURIComponent(id)),id));
const links=await page.locator('a[href^="/"]').evaluateAll(as=>[...new Set(as.map(a=>a.getAttribute('href')))]);
for(const link of links){const r=await page.request.get(new URL(link,url).href);check('local link '+link,r.status()===200);}
for(const [section,name]of [['#matrix','checked-download.csv'],['#sixteen','checked-16gpu.csv']]){const download=page.waitForEvent('download');await page.locator(section+' > p > a[download]').click();const d=await download;check('CSV download '+section,(await d.suggestedFilename()).endsWith('.csv'));await d.saveAs(path.join(out,name));}
await page.locator('#full-details > summary').click();await page.goto(url);await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
await page.setViewportSize({width:390,height:844});await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
check('no removed diagram headings',await page.getByRole('heading',{name:/图 1|图 2/}).count()===0);
check('mobile no page overflow',await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
check('no JS errors',errors.length===0);
}finally{await browser.close();fs.writeFileSync(path.join(out,'web_acceptance.json'),JSON.stringify({passed:checks.every(c=>c.passed),checks},null,2));}
console.log(`${checks.length} browser checks PASS`);
})().catch(e=>{console.error(e);process.exit(1)});
