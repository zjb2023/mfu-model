const {chromium}=require('playwright'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
(async()=>{const out=process.argv[2];fs.mkdirSync(out,{recursive:false});const b=await chromium.launch({executablePath:'/home/zjb/.cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell',args:['--no-sandbox']});try{
 const p=await b.newPage({viewport:{width:1500,height:1100}}),errors=[];p.on('pageerror',e=>errors.push(String(e)));
 assert.equal((await p.goto('http://127.0.0.1:43312/df-v001/detailed-fb-32to256-r1/')).status(),200);
 assert.match(await p.locator('#result').innerText(),/6\.16/);assert.equal(await p.locator('#pipeline rect').count(),128);assert.equal(await p.locator('#comparison rect').count(),16);
 await p.locator('#pipeline rect').nth(8).click({force:true});assert.match(await p.locator('#detail').innerText(),/s1:F0/);
 assert(await p.evaluate(()=>[...document.querySelectorAll('rect')].every(r=>Number(r.getAttribute('width'))>0)));
 await p.locator('#comparison').screenshot({path:path.join(out,'comparison.png')});await p.locator('#pipeline').screenshot({path:path.join(out,'pipeline.png')});
 await p.setViewportSize({width:390,height:844});assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));assert.deepEqual(errors,[]);
 fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify({status:'PASS',checks:['HTTP200','6.16% report','128 predicted blocks','16 prediction/trace blocks','node click','positive widths','mobile no page overflow','no JS errors']}));console.log('PASS 256 UI');
}finally{await b.close()}})().catch(e=>{console.error(e);process.exit(1)});
