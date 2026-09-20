const {chromium}=require('playwright');
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
(async()=>{
 const out=process.argv[2];fs.mkdirSync(out,{recursive:false});
 const browser=await chromium.launch({executablePath:'/home/zjb/.cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell',args:['--no-sandbox']});
 try{
  const p=await browser.newPage({viewport:{width:1400,height:1000}}),errors=[];
  p.on('pageerror',e=>errors.push(String(e)));
  const base='http://127.0.0.1:43312/df-v001/';
  assert.equal((await p.goto(base+'cp-ep8-structure-r1/#b-semantics')).status(),200);
  assert.match(await p.locator('#b-semantics').innerText(),/288/);
  assert.equal(await p.locator('#b-semantics .comm').count(),9);
  assert.equal(await p.locator('#logic-frame').count(),1);
  await p.waitForFunction(()=>document.querySelector('#result').textContent.includes('84.864'));
  await p.locator('#epplus').click();
  assert.match(await p.locator('#result').innerText(),/85\.864/);
  await p.locator('#reset').click();
  assert.match(await p.locator('#result').innerText(),/84\.864/);
  await p.locator('#b-semantics').screenshot({path:path.join(out,'b-logic.png')});
  assert.equal((await p.goto(base+'b-cp-identity-r1/')).status(),200);
  assert.equal(await p.locator('tbody tr').count(),288);
  assert.equal(await p.locator('details').count(),8);
  const resp=await p.request.get(base+'b-cp-identity-r1/events.json');
  assert.equal((await resp.json()).length,288);
  await p.setViewportSize({width:390,height:844});
  await p.goto(base+'cp-ep8-structure-r1/#b-semantics');
  assert(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
  assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(out,'checks.json'),JSON.stringify({status:'PASS',checks:['9 CP slots shown','288 evidence rows','8 ranks','F logic retained','Combine +1 and reset regression','mobile no page overflow','no JS errors']},null,2));
  console.log('PASS B CP evidence UI and F cost regression');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
