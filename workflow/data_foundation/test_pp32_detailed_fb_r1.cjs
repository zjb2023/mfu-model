const {chromium}=require('playwright'),fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
(async()=>{const out=process.argv[2];fs.mkdirSync(out,{recursive:false});const b=await chromium.launch({executablePath:'/home/zjb/.cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell',args:['--no-sandbox']});try{
 const p=await b.newPage({viewport:{width:1500,height:1000}}),errors=[];p.on('pageerror',e=>errors.push(String(e)));
 assert.equal((await p.goto('http://127.0.0.1:43312/df-v001/pp32-detailed-fb-r1/')).status(),200);
 assert.match(await p.locator('#report').innerText(),/11917\.23/);
 for(const it of ['60','70']){await p.selectOption('#iteration',it);assert.equal(await p.locator('#timeline rect').count(),128);assert(await p.evaluate(()=>[...document.querySelectorAll('#timeline rect')].every(r=>Number(r.getAttribute('width'))>0)))}
 await p.screenshot({path:path.join(out,'desktop.png'),fullPage:true});await p.locator('#timeline').screenshot({path:path.join(out,'timeline.png')});
 await p.setViewportSize({width:390,height:844});assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));assert.deepEqual(errors,[]);
 fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify({status:'PASS',checks:['HTTP200','prediction value','60/70 switch','128 positive FB rectangles','mobile no page overflow','no JS errors']}));console.log('PASS assembly UI');
}finally{await b.close()}})().catch(e=>{console.error(e);process.exit(1)});
