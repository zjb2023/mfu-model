const fs=require('fs'),path=require('path');
const root=path.resolve(__dirname,'../../../..'),out=path.join(root,'results/w37/A/demo4'),docs=path.join(root,'docs/w37/1f1b');
const {chromium}=require(path.join(root,'results/w37/A/research-html-20260907/browser-check/node_modules/playwright'));
const checks=[];function check(name,v){checks.push({name,passed:!!v});if(!v)throw Error(name)}
(async()=>{const browser=await chromium.launch({headless:true,executablePath:'/home/zjb/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome',args:['--no-sandbox']});try{
const p=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];p.on('pageerror',e=>errors.push(e.message));
const r=await p.goto('http://192.168.0.49:8037/docs/w37/1f1b/DAG_4GPU_DEMO.html');check('HTTP UTF8',r.status()===200&&r.headers()['content-type'].includes('utf-8'));
let d=await p.evaluate(()=>window.demoState);check('33 nodes / 56 edges',d.nodes.length===33&&d.edges.length===56);check('hand calculated default DAG34 phase36',d.total===34&&d.phase.total===36);
check('formula parameters visible',await p.locator('#phase-formula tbody tr').count()===9);
check('default formula substitution',(await p.locator('#phase-substitution').innerText()).includes('整轮 = 17 + 18 + 1 = 36 ms'));
check('F/B overlap',Math.min(...d.nodes.filter(n=>n.kind==='B').map(n=>n.start))<Math.max(...d.nodes.filter(n=>n.kind==='F').map(n=>n.end)));
for(let slow=2;slow<=6;slow++)for(let comm=0;comm<=3;comm++){
await p.evaluate(({slow,comm})=>{document.getElementById('slow').value=slow;document.getElementById('comm').value=comm;document.getElementById('slow').dispatchEvent(new Event('input'));},{slow,comm});d=await p.evaluate(()=>window.demoState);
check('formula substitution updates '+slow+','+comm,(await p.locator('#phase-substitution').innerText()).includes('整轮 = '+d.phase.F+' + '+d.phase.B+' + 1 = '+d.phase.total+' ms'));
check('all predecessor conditions '+slow+','+comm,d.nodes.every(n=>n.start===Math.max(0,...n.pred.map(id=>d.by[id].end))&&n.end===n.start+n.duration));
check('resource exclusivity '+slow+','+comm,[...new Set(d.nodes.map(n=>n.lane))].every(lane=>{const ns=d.nodes.filter(n=>n.lane===lane&&n.duration>0).sort((a,b)=>a.start-b.start);return ns.every((n,i)=>i===0||n.start>=ns[i-1].end);}));
}
await p.locator('#uniform').click();d=await p.evaluate(()=>window.demoState);check('uniform hand solution 32=32',d.total===32&&d.phase.total===32);
await p.evaluate(()=>{document.getElementById('comm').value=0;document.getElementById('comm').dispatchEvent(new Event('input'));});d=await p.evaluate(()=>window.demoState);check('zero network homogeneous 26',d.total===26&&d.phase.total===26);
await p.locator('#reset').click();await p.locator('[data-node="s1:B0"]').click();check('node why wait 21ms',(await p.locator('#inspection').innerText()).includes('21 ms'));
for(const id of await p.locator('[data-node]').evaluateAll(es=>es.map(e=>e.dataset.node))){await p.locator('[data-node="'+id+'"]').click();check('node inspection '+id,(await p.locator('#inspection').innerText()).includes('成本'));}
await p.locator('#edges').uncheck();await p.locator('#critical').uncheck();check('all dependency paths hidden',await p.locator('#dag > path').count()===0);await p.locator('#edges').check();await p.locator('#critical').check();check('all edges restored',await p.locator('#dag > path').count()===56);
await p.locator('#reset').click();await p.locator('[data-node="s1:B0"]').click();
for(const [id,name] of [['svg-download','DAG_4GPU_DEMO.svg'],['json-download','DAG_4GPU_DEMO.nodes_edges.json']]){const downloaded=p.waitForEvent('download');await p.locator('#'+id).click();const f=await downloaded;await f.saveAs(path.join(docs,name));check('export '+name,fs.statSync(path.join(docs,name)).size>1000);}
await p.locator('#dag').screenshot({path:path.join(docs,'DAG_4GPU_DEMO.png')});await p.screenshot({path:path.join(out,'desktop.png')});
await p.setViewportSize({width:390,height:844});await p.screenshot({path:path.join(out,'mobile.png')});check('mobile horizontal overflow isolated',await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
await p.goto('file://'+path.join(docs,'DAG_4GPU_DEMO.html'));await p.locator('#uniform').click();check('standalone offline interactivity',(await p.evaluate(()=>window.demoState.total))===32);
check('no browser errors',errors.length===0);
}finally{await browser.close();fs.writeFileSync(path.join(out,'acceptance.json'),JSON.stringify({passed:checks.every(c=>c.passed),checks},null,2));}console.log(checks.length+' teaching DAG checks PASS');})().catch(e=>{console.error(e);process.exit(1)});
