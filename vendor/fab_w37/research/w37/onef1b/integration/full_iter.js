const fullIterData=JSON.parse(document.getElementById('full-iter-data').textContent);
const svgNS='http://www.w3.org/2000/svg';
const fiFormat=(v,n=3)=>Number(v).toFixed(n);
function fiSvg(parent,tag,attrs={},text=''){
  const e=document.createElementNS(svgNS,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text)e.textContent=text;parent.append(e);return e;
}
function fiDetail(view,b){
  const name=b.phase?(b.phase==='forward'?'前向 F':'反向 B')+b.microbatch:(b.group_type==='optimizer'?'参数更新 / 运行时':b.group_type==='dp'?'DP 通信':'EDP 通信')+' · '+b.collective;
  return `${view.label} · PP${b.stage} · ${name} · ${fiFormat(b.start_ms/1000,6)}–${fiFormat(b.end_ms/1000,6)} 秒 · 区间 ${fiFormat(b.end_ms-b.start_ms,6)} ms · ${b.scope}`;
}
function drawFullIter(view,range,showComms){
  const block=document.createElement('figure');block.className='full-iter-figure';block.dataset.view=view.id;
  const title=document.createElement('h3');title.textContent=view.label;block.append(title);
  const totals=document.createElement('p');totals.className='muted';totals.textContent=`Profiler ${fiFormat(view.profiler_ms/1000)} 秒；outer（两种计时口径的总时长差）${fiFormat(view.outer_ms/1000)} 秒；Training 整轮 ${fiFormat(view.training_ms/1000)} 秒。`+(view.id==='prediction'?'预测固定，不随选择的观测轮次重新拟合。':'');block.append(totals);
  const accounting=document.createElement('div');accounting.className='note full-iter-accounting';
  if(view.id==='prediction'){
    const formula=document.createElement('p');formula.textContent=`时间总账：原始图 ${fiFormat(view.raw_graph_ms/1000,6)} 秒 + 图→Profiler 补差 ${fiFormat(view.reconciliation_ms,3)} ms → Profiler ${fiFormat(view.profiler_ms/1000,6)} 秒；再加预测 outer ${fiFormat(view.outer_ms,3)} ms → Training ${fiFormat(view.training_ms/1000,6)} 秒。`;accounting.append(formula);
  }
  const scope=document.createElement('p');scope.textContent='本图横轴使用 Profiler 口径，outer 未画成时间条，也未均分到 F/B 或其他节点。现有证据只给出总时长差，尚不能确定它对应哪些事件、发生在什么位置。'+(view.id==='prediction'?'图→Profiler 的 604.440 ms 同样只有总量；它与 outer 是两项不同的补差，不能把它们视为同一开销。':'观测 outer = 本轮训练日志时间 − 本轮 Profiler 时间。');accounting.append(scope);block.append(accounting);
  const frame=document.createElement('div');frame.className='full-iter-frame';block.append(frame);
  const W=1440,left=100,right=35,top=115,rh=72,H=top+view.stages*rh+65;
  const root=fiSvg(frame,'svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':view.label+'完整迭代时间图'});
  const x=v=>left+(v-range[0])/(range[1]-range[0])*(W-left-right);
  fiSvg(root,'rect',{width:W,height:H,fill:'#fff'});
  const id='fi-clip-'+view.id;const defs=fiSvg(root,'defs');const clip=fiSvg(defs,'clipPath',{id});fiSvg(clip,'rect',{x:left,y:top-5,width:W-left-right,height:view.stages*rh+6});
  const band=(a,b,color,label)=>{
    const aa=Math.max(range[0],a),bb=Math.min(range[1],b);if(bb<=aa)return;
    const r=fiSvg(root,'rect',{x:x(aa),y:42,width:x(bb)-x(aa),height:26,fill:color});fiSvg(r,'title',{},`${label}：${fiFormat(a/1000)}–${fiFormat(b/1000)} 秒`);
    if(x(bb)-x(aa)>110)fiSvg(root,'text',{x:(x(aa)+x(bb))/2,y:60,'text-anchor':'middle','font-size':14,fill:'#173249'},label);
  };
  fiSvg(root,'text',{x:12,y:60,'font-size':13,fill:'#4c6376'},'全局分段');
  band(0,view.entry_ms,'#f0d79e','首 F 前准备');band(view.entry_ms,view.fb_end_ms,'#daedf4','完整 F/B 区间');band(view.fb_end_ms,view.profiler_ms,'#d7cbe6',view.id==='prediction'?'收尾总账（含补差）':'最后 B 后收尾');
  fiSvg(root,'text',{x:left,y:91,'font-size':13,fill:'#4c6376'},'每个 PP 行依次：F / B / DP / EDP / OPT；留白表示本图未展示活动，不能据此认定 GPU 空闲。');
  const body=fiSvg(root,'g',{'clip-path':`url(#${id})`});
  for(let stage=0;stage<view.stages;stage++){
    const y=top+stage*rh;fiSvg(body,'rect',{x:left,y,width:W-left-right,height:rh,fill:stage%2?'#f5f8fa':'#edf3f7'});
    fiSvg(root,'text',{x:10,y:y+22,'font-size':15,'font-weight':650,fill:'#173249'},'PP'+stage);
    for(const [text,offset]of [['F',10],['B',27],['DP',43],['EDP',54],['OPT',65]])fiSvg(root,'text',{x:left-8,y:y+offset,'text-anchor':'end','font-size':9,fill:'#647586'},text);
    if(view.id!=='prediction'&&showComms)fiSvg(body,'text',{x:left+5,y:y+65,'font-size':9,fill:'#82909d'},'优化器计算明细未单列');
  }
  const tick=(range[1]-range[0])>15000?5000:2000;
  for(let t=Math.ceil(range[0]/tick)*tick;t<=range[1];t+=tick){fiSvg(root,'line',{x1:x(t),y1:top-8,x2:x(t),y2:H-48,stroke:'#cbd7df','stroke-dasharray':'3 4'});fiSvg(root,'text',{x:x(t),y:H-26,'text-anchor':'middle','font-size':13,fill:'#40596b'},fiFormat(t/1000,0)+' 秒');}
  const paint=(b,y,height,color)=>{
    if(b.end_ms<range[0]||b.start_ms>range[1])return;
    const r=fiSvg(body,'rect',{x:x(b.start_ms),y,width:Math.max(1.5,x(b.end_ms)-x(b.start_ms)),height,rx:2,fill:color,class:'full-iter-event','data-kind':b.phase||b.group_type,tabindex:0,role:'button','aria-label':fiDetail(view,b)});
    fiSvg(r,'title',{},fiDetail(view,b));const select=()=>document.getElementById('full-iter-selection').textContent=fiDetail(view,b);
    r.addEventListener('click',select);r.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select()}});
    if(b.phase&&x(b.end_ms)-x(b.start_ms)>24)fiSvg(body,'text',{x:x(b.start_ms)+3,y:y+10,'font-size':10,fill:'#fff','pointer-events':'none'},(b.phase==='forward'?'F':'B')+b.microbatch);
  };
  for(const b of view.bars)paint(b,top+b.stage*rh+(b.phase==='forward'?0:17),13,b.phase==='forward'?'#1488b3':'#ce6045');
  if(showComms)for(const b of view.collectives){const opt=b.group_type==='optimizer',dp=b.group_type==='dp';paint(b,top+b.stage*rh+(opt?59:dp?37:48),7,opt?'#8e6613':dp?'#237650':'#7350a8');}
  for(const [v,color,dash]of [[view.entry_ms,'#ab7b23','3 5'],[view.fb_end_ms,'#88709d','3 5'],[view.profiler_ms,'#173249','']])if(v>=range[0]&&v<=range[1])fiSvg(root,'line',{x1:x(v),y1:32,x2:x(v),y2:H-48,stroke:color,'stroke-dasharray':dash,'stroke-width':1.5});
  fiSvg(root,'text',{x:(left+W-right)/2,y:H-5,'text-anchor':'middle','font-size':13,fill:'#4c6376'},'相对各自 Profiler 迭代起点的秒数（统一比例）');
  return block;
}
function renderFullIter(){
  const it=document.getElementById('full-iter-choice').value;
  const views=[fullIterData.prediction,fullIterData.trace[it].target,fullIterData.trace[it].source];
  const window=document.getElementById('full-iter-window').value,show=document.getElementById('full-iter-comms').checked;
  const max=Math.ceil(Math.max(...views.map(v=>v.profiler_ms))/5000)*5000;
  const range=window==='all'?[0,max]:window==='fb'?[Math.min(...views.map(v=>v.entry_ms)),Math.max(...views.map(v=>v.fb_end_ms))]:[Math.max(0,Math.min(...views.map(v=>v.fb_end_ms))-750),max];
  document.getElementById('full-iter-plots').replaceChildren(...views.map(v=>drawFullIter(v,range,show)));
  const body=document.querySelector('#full-iter-ledger tbody');body.replaceChildren();
  for(const v of views){const tr=document.createElement('tr');for(const val of [v.label,v.entry_ms,v.fb_end_ms-v.entry_ms,v.profiler_ms-v.fb_end_ms,v.profiler_ms,v.outer_ms,v.training_ms]){const td=document.createElement('td');td.textContent=typeof val==='number'?fiFormat(val/1000):val;tr.append(td)}body.append(tr);}
  const p=views[0],t=views[1],ape=(a,b)=>Math.abs(a-b)/b*100;
  document.getElementById('full-iter-score').textContent=`第 ${it} 轮目标开发评估：Profiler APE ${fiFormat(ape(p.profiler_ms,t.profiler_ms))}%；Training APE ${fiFormat(ape(p.training_ms,t.training_ms))}%；1F1B APE ${fiFormat(ape(p.fb_end_ms-p.entry_ms,t.fb_end_ms-t.entry_ms))}%。这些误差仍包含其定义对应的完整区间；75.8% 属于后文 v682 的历史分账。`;
  document.getElementById('full-iter-selection').textContent='点击图中彩条查看详情。';
}
for(const id of ['full-iter-choice','full-iter-window','full-iter-comms'])document.getElementById(id).addEventListener('change',renderFullIter);
renderFullIter();
