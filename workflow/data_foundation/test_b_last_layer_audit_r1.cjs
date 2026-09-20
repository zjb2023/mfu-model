const {chromium}=require('playwright'),fs=require('node:fs'),assert=require('node:assert/strict'),path=require('node:path');
(async()=>{const out=process.argv[2];fs.mkdirSync(out,{recursive:false});const b=await chromium.launch({executablePath:'/home/zjb/.cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell',args:['--no-sandbox']});try{
 const p=await b.newPage({viewport:{width:1440,height:1100}}),errors=[];p.on('pageerror',e=>errors.push(String(e)));
 assert.equal((await p.goto('http://127.0.0.1:43312/df-v001/b-r10-60to70-r3/#last-layer-audit')).status(),200);
 assert.equal(await p.locator('#last-layer-audit table').count(),4);assert.match(await p.locator('#last-layer-audit').innerText(),/3\.829/);
 for(let r=8;r<16;r++){await p.selectOption('#rank',String(r));assert.equal(await p.locator('#phases tr').count(),8)}
 await p.locator('#last-layer-audit').screenshot({path:path.join(out,'audit.png')});
 await p.setViewportSize({width:390,height:844});assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));assert.deepEqual(errors,[]);
 fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify({status:'PASS',checks:['HTTP200','four audit tables','8 rank controls preserved','mobile no page overflow','no JS errors']}));console.log('PASS audit UI');
}finally{await b.close()}})().catch(e=>{console.error(e);process.exit(1)});
