const {chromium}=require('/home/zjb/Desktop/worktrees/mfu-w37-v610/node_modules/playwright');
(async()=>{
 const b=await chromium.launch({headless:true,executablePath:'/home/zjb/.cache/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-linux64/chrome-headless-shell'});
 try {
  const p=await b.newPage({viewport:{width:1440,height:1000}}),errors=[];p.on('pageerror',e=>errors.push(e.message));
  await p.goto('http://192.168.8.16:43312/df-v001/b-recompute-r11/');await p.waitForSelector('#recompute-moe .dag-node');
  const baseline=await p.evaluate(()=>current.END.end_ms);
  if(Math.abs(baseline-868.8333587646489)>1e-6)throw Error('baseline mismatch');
  for(let u=0;u<4;u++){await p.locator('#layers button').nth(u).click();if(await p.locator('#recompute-attention .dag-node').count()!==32||await p.locator('#recompute-moe .dag-node').count()!==34)throw Error('missing anchors');}
  await p.locator('#layers button').nth(0).click();
  for(const [selector,increment] of [['#recompute-moe .dag-node[aria-label="FC1 前向计算"]',100],['#recompute-attention .dag-node[aria-label="CP0 · 8/9"]',100],['#recompute-moe .dag-node[aria-label="EP8合并有效残余"]',1]]){
   await p.locator(selector).first().click();const cost=Number(await p.locator('#dag-cost').inputValue());await p.locator('#dag-cost').fill(String(cost+increment));await p.locator('#dag-apply').click();
   const changed=await p.evaluate(()=>current.END.end_ms);if(changed<=baseline)throw Error('no end-to-end propagation');
   console.log(selector,'increment',increment,'B shift',changed-baseline);
   await p.locator('#reset').click();if(Math.abs(await p.evaluate(()=>current.END.end_ms)-baseline)>1e-6)throw Error('reset failed');
  }
  await p.locator('#rank').selectOption('15');await p.setViewportSize({width:390,height:844});
  if(await p.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('page overflow');
  if(errors.length)throw Error(errors.join('\n'));
  console.log('PASS four layers, compute/CP/EP edits, reset, rank switch, mobile width, no JS errors');
 } finally {await b.close();}
})().catch(e=>{console.error(e);process.exit(1)});
