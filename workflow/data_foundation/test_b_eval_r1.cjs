const {chromium}=require('playwright'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
 const out=process.argv[2];fs.mkdirSync(out,{recursive:false});
 const browser=await chromium.launch({executablePath:'/home/zjb/.cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell',args:['--no-sandbox']});
 try{
  const p=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];p.on('pageerror',e=>errors.push(String(e)));
  const response=await p.goto('http://127.0.0.1:43312/df-v001/b-r10-60to70-r2/');assert.equal(response.status(),200);
  assert.match(await p.locator('#summary').innerText(),/893\.788/);
  assert.equal(await p.locator('#timeline rect').count(),16);
  for(let rank=8;rank<16;rank++){
   await p.selectOption('#rank',String(rank));assert.equal(await p.locator('#phases tr').count(),8);assert.equal(await p.locator('#components tr').count(),40);
   assert(await p.evaluate(()=>[...document.querySelectorAll('#timeline rect')].every(r=>Number(r.getAttribute('width'))>0)));
  }
  await p.selectOption('#rank','8');await p.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
  await p.setViewportSize({width:390,height:844});assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  assert.deepEqual(errors,[]);fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify({status:'PASS',checks:['HTTP 200','8 rank selections','64 phase records','320 component records','positive timeline widths','mobile no page overflow','no JS errors']}));
  console.log('PASS B evaluation UI');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
