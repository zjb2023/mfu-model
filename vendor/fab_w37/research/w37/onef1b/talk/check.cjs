const fs=require('fs'),path=require('path');
const root=path.resolve(__dirname,'../../../..'),out=path.join(root,'results/w37/A/talk-20260908');
const {chromium}=require(path.join(root,'results/w37/A/research-html-20260907/browser-check/node_modules/playwright'));
const checks=[];function check(name,p){checks.push({name,passed:!!p});if(!p)throw Error(name)}
(async()=>{const browser=await chromium.launch({headless:true,executablePath:'/home/zjb/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome',args:['--no-sandbox']});
try{
const p=await browser.newPage({viewport:{width:1360,height:1000}});const errors=[];p.on('pageerror',e=>errors.push(e.message));
const url='http://192.168.0.49:8037/docs/w37/1f1b/TALK_MFU_EXTRAPOLATION.html';const r=await p.goto(url);check('UTF8 HTTP200',r.status()===200&&r.headers()['content-type'].includes('utf-8'));
check('five presentation stops',await p.locator('table').first().locator('tbody tr').count()===5);
for(const text of ['10.561626%','10.043845%','11.167230%','45.7773%','44.8083%','不是现在已经验证的能力','这轮不开额外DeepEP日志'])check('reviewed statement '+text,(await p.locator('main').innerText()).includes(text));
await p.locator('summary').click();check('table of contents opens',await p.locator('details').getAttribute('open')!==null);
const local=await p.locator('a[href^="#"]').evaluateAll(es=>es.map(e=>e.getAttribute('href').slice(1)));for(const id of local)check('talk anchor '+id,await p.evaluate(id=>!!document.getElementById(decodeURIComponent(id)),id));
const links=await p.locator('a:not([href^="#"])').evaluateAll(es=>[...new Set(es.map(e=>e.href))]);
for(const u of links){const res=await p.request.get(u);check('link '+new URL(u).pathname,res.status()===200);if(new URL(u).hash){const q=await browser.newPage();await q.goto(u);const id=decodeURIComponent(new URL(u).hash.slice(1));check('presentation anchor '+id,await q.evaluate(id=>!!document.getElementById(id),id));await q.close();}}
await p.evaluate(()=>window.print=()=>window.printInvoked=true);await p.locator('#print').click();check('print button',await p.evaluate(()=>window.printInvoked===true));
await p.locator('summary').click();await p.screenshot({path:path.join(out,'desktop.png')});await p.setViewportSize({width:390,height:844});await p.screenshot({path:path.join(out,'mobile.png')});check('mobile width',await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));check('no JS errors',errors.length===0);
}finally{await browser.close();fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify({passed:checks.every(c=>c.passed),checks},null,2));}
console.log(checks.length+' talk checks PASS');})().catch(e=>{console.error(e);process.exit(1)});
