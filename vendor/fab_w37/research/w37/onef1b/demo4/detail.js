(()=>{
const $=id=>document.getElementById(id),colors={F:'#287eb4',B:'#cd8438',TP:'#8966b4',PP:'#008b91',OPT:'#398252'};
function build(cost){
 const nodes=[],by={};
 function add(id,name,kind,lane,duration,pred){const n={id,name,kind,lane,duration,pred:[...new Set(pred.filter(Boolean))]};nodes.push(n);by[id]=n;return id;}
 const ends={},lastGPU={},lastTP={},lastPP={},lastAction={};
 for(let s=0;s<2;s++)for(const [phase,m] of window.tpOrders[s]){
  let prior=[lastAction[s],...[0,1].map(l=>lastGPU[s*2+l])];
  if(phase==='F'&&s===1)prior=prior.concat([`ppF${m}:0`,`ppF${m}:1`]);
  if(phase==='B'){prior=prior.concat([`end${s}:F${m}`]);if(s===0)prior=prior.concat([`ppB${m}:0`,`ppB${m}:1`]);}
  for(const local of phase==='F'?[0,1]:[1,0])for(const mod of phase==='F'?['A','M']:['M','A']){
   const layer=s*2+local,pair=[];
   for(let l=0;l<2;l++){
    const rank=s*2+l,id=`r${rank}:${phase}:m${m}:L${layer}:${mod}`;
    let duration=phase==='F'?2:3;
    if(s===0&&phase==='F'&&m===1&&layer===0&&mod==='A')duration=l===0?cost:4;
    pair.push(add(id,`GPU${rank} / PP${s} / MB${m} / 层${layer} / ${mod==='A'?'Attention':'MLP'}${phase==='F'?'前向':'反向'}`,phase,rank,duration,prior.concat([lastGPU[rank]])));lastGPU[rank]=id;
   }
   const id=`tp${s}:${phase}:m${m}:L${layer}:${mod}`;
   add(id,`PP${s}两张GPU的TP通信 / MB${m} / 层${layer} / ${phase}${mod}`,'TP',4+s,1,pair.concat([lastTP[s]]));lastTP[s]=id;prior=[id];
  }
  const end=`end${s}:${phase}${m}`;add(end,`PP${s} / ${phase}${m}完成`,'gate',-1,0,prior);ends[`${s}:${phase}${m}`]=end;lastAction[s]=end;
  if((phase==='F'&&s===0)||(phase==='B'&&s===1))for(let l=0;l<2;l++){
   const id=`pp${phase}${m}:${l}`,key=phase+l;
   add(id,`${phase==='F'?'激活 PP0→PP1':'梯度 PP1→PP0'} / MB${m} / TP lane${l}`,'PP',6,1,[end,lastPP[key]]);lastPP[key]=id;
  }
 }
 for(let r=0;r<4;r++)add(`opt${r}`,`GPU${r} 参数更新`,'OPT',r,1,[ends['0:B1'],ends['1:B1']]);
 let pending=nodes.slice();while(pending.length){let done=0;pending=pending.filter(n=>{if(n.pred.some(id=>by[id].end===undefined))return true;n.start=Math.max(0,...n.pred.map(id=>by[id].end));n.end=n.start+n.duration;done++;return false;});if(!done)throw Error('Cycle in teaching DAG');}
 return {nodes,by,total:Math.max(...nodes.map(n=>n.end)),cost,edges:nodes.flatMap(n=>n.pred.map(id=>({src:id,dst:n.id}))) };
}
const base=build(2);let current,selected='r0:F:m1:L0:A';
function inspect(){const n=current.by[selected];$('tp-inspect').innerHTML=`<b>${n.name}</b><p>开始 ${n.start} ms + 自身耗时 ${n.duration} ms = 结束 ${n.end} ms；基准结束 ${base.by[n.id].end} ms。</p><p>开始时间取以下前驱完成时间的最大值：${n.pred.map(id=>current.by[id].name+'（'+current.by[id].end+' ms）').join('；')||'无前驱，从0开始'}。</p>`;}
function draw(){
 const cost=+$('tp-cost').value;current=build(cost);window.tpDemoState=current;$('tp-cost-value').textContent=cost;
 const delta=current.total-base.total,local=current.by['tp0:F:m1:L0:A'].end-base.by['tp0:F:m1:L0:A'].end;
 $('tp-summary').textContent=`基准整轮 ${base.total} ms → 当前 ${current.total} ms（增加 ${delta} ms）。选中计算增加 ${cost-2} ms，紧随其后的TP同步完成推迟 ${local} ms。${local===0?'增加的计算耗时被原有等待吸收；整轮时间不变。':delta===0?'同步及部分后续任务虽已推迟，但下游仍在处理先前微批；整轮关键路径未变。红框不等于整轮误差贡献。':'延迟超过可被下游等待吸收的范围，整轮也变慢；红框标出结束时间推迟的任务。'}固定有效FLOPs与硬件峰值时，当前MFU / 基准MFU = ${base.total}/${current.total} = ${(base.total/current.total).toFixed(4)}；没有实测真值，不称为预测误差。`;
 const max=Math.max(base.total,current.total)+2,x=t=>150+t/max*1230,y=lane=>55+[0,1,4,5,2,6,3][lane]*70;
 let svg='<defs><marker id="tp-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#668397"/></marker><marker id="tp-arrow-focus" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 Z" fill="#123f70"/></marker></defs><rect width="1420" height="600" fill="white"/>';
 for(let t=0;t<=max;t+=5)svg+=`<line x1="${x(t)}" x2="${x(t)}" y1="30" y2="560" stroke="#e5edf1"/><text x="${x(t)}" y="22" font-size="12">${t} ms</text>`;
 const names=['GPU0 · PP0','GPU1 · PP0','GPU2 · PP1','GPU3 · PP1','TP · GPU0/1','TP · GPU2/3','PP · 两条lane'];
 names.forEach((v,i)=>{svg+=`<text x="6" y="${y(i)+22}" font-size="14">${v}</text>`;});
 // Contract zero-duration completion gates for display, then remove transitive edges only in this view.
 const ancestors={};function before(id){if(ancestors[id])return ancestors[id];const a=new Set();for(const p of current.by[id].pred){a.add(p);for(const q of before(p))a.add(q);}return ancestors[id]=a;}
 function visiblePred(id,via=false){const n=current.by[id];return n.lane>=0?[{id,via}]:n.pred.flatMap(p=>visiblePred(p,true));}
 const shown=[];
 for(const n of current.nodes.filter(n=>n.lane>=0)){
  const ps=[...new Map(n.pred.flatMap(p=>visiblePred(p)).map(p=>[p.id,p])).values()];
  for(const p of ps)if(!ps.some(q=>q.id!==p.id&&before(q.id).has(p.id)))shown.push({src:p.id,dst:n.id,via:p.via});
 }
 window.tpDisplayEdges=shown;
 const mode=$('tp-edge-mode').value;
 function geometry(n){const pp=n.kind==='PP';return {top:y(n.lane)+(pp&&n.id.endsWith(':1')?24:0),h:pp?20:34};}
 function arrow(e){
  const p=current.by[e.src],n=current.by[e.dst],pg=geometry(p),ng=geometry(n),focus=e.src===selected||e.dst===selected;
  if(mode==='none'||mode==='local'&&!focus)return '';
  const a=x(p.end),b=x(n.start),down=ng.top>pg.top,sy=pg.top+(down?pg.h:0),ty=ng.top+(down?0:ng.h);
  // Use a small left detour at equal-time boundaries so the arrowhead remains visible outside the target.
  const d=pg.top===ng.top?`M${a-2} ${pg.top} C${a} ${pg.top-15} ${b+2} ${ng.top-15} ${b+2} ${ng.top}`:`M${a-2} ${sy} C${a-5} ${(sy+ty)/2} ${b+2} ${(sy+ty)/2} ${b+2} ${ty}`;
  return `<path data-tp-edge="${e.src}>${e.dst}" data-focus="${focus}" d="${d}" stroke="${focus?'#123f70':'#668397'}" stroke-width="${focus?2.4:1}" stroke-opacity="${focus?1:0.38}" ${e.via?'stroke-dasharray="4 3"':''} marker-end="url(#${focus?'tp-arrow-focus':'tp-arrow'})" fill="none" pointer-events="none"><title>${p.name} → ${n.name}${e.via?'（经零耗时完成点）':''}</title></path>`;
 }
 // Draw ordinary arrows first and selected dependencies last, both behind task blocks.
 svg+=shown.filter(e=>e.src!==selected&&e.dst!==selected).map(arrow).join('');
 svg+=shown.filter(e=>e.src===selected||e.dst===selected).map(arrow).join('');
 for(const n of current.nodes.filter(n=>n.duration>0)){
  const delayed=n.end>base.by[n.id].end,pp=n.kind==='PP',offset=pp&&n.id.endsWith(':1')?24:0,h=pp?20:34;
  svg+=`<g data-tp-node="${n.id}" tabindex="0" role="button" aria-label="${n.name}"><title>${n.name}：${n.start}→${n.end} ms</title><rect x="${x(n.start)}" y="${y(n.lane)+offset}" width="${Math.max(3,x(n.end)-x(n.start))}" height="${h}" fill="${colors[n.kind]}" stroke="${n.id===selected?'#111':delayed?'#da3434':'white'}" stroke-width="${n.id===selected||delayed?3:1}"/><text x="${x(n.start)+2}" y="${y(n.lane)+offset+14}" fill="white" font-size="8">${n.kind==='F'||n.kind==='B'?n.id.split(':').slice(1).join(' '):n.kind}</text></g>`;
 }
 $('tp-chart').innerHTML=svg;document.querySelectorAll('[data-tp-node]').forEach(el=>{el.onclick=()=>{selected=el.dataset.tpNode;draw();};el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();el.onclick();}};});inspect();
}
$('tp-edge-mode').onchange=draw;$('tp-cost').oninput=draw;for(const [id,c] of [['tp-hidden',3],['tp-local',6],['tp-exposed',24],['tp-reset',2]])$(id).onclick=()=>{$('tp-cost').value=c;draw();};
$('tp-export').onclick=()=>{const u=URL.createObjectURL(new Blob([JSON.stringify(current,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download='4GPU_PP2_TP2_DAG.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);};draw();
})();
