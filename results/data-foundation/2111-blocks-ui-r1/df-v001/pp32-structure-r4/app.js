'use strict';
const $=id=>document.getElementById(id);let data,by,selected=null;
const color={F:'#dceaf9',B:'#dff1eb',PP:'#eee5f6'};
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function details(n){selected=n.id;$('selectedTitle').textContent=n.id+' · '+n.kind;const p=data.parameters[n.parameter];
 $('inspect').innerHTML=`<p>参与rank：${n.members.join(', ')} · ${escape(n.status)}</p><p>参数：${escape(n.parameter||'无耗时辅助节点')} ${p?' / '+(p.value_ms===null?'待测':p.value_ms.toFixed(3)+' ms（旧试点值）'):''}</p><p>完整图中的直接前驱（包括当前视图隐藏的微批和就绪节点）：</p><div class="preds">${n.predecessors.map(k=>'<code>'+escape(k)+'</code>').join('')}</div><p>${escape(n.note||'按源CPU调度与微批编号建立依赖；不代表独立GPU因果验证。')}</p>`;
}
function seq(s){const w=3-s,a=[];for(let m=0;m<w;m++)a.push('F'+m);for(let j=0;j<8-w;j++)a.push('F'+(w+j),'B'+j);for(let m=8-w;m<8;m++)a.push('B'+m);return a;}
function draw(){const lane=+$('chain').value,all=$('allRanks').checked,mb=$('micro').value,many=mb==='all';const ranks=all?Array.from({length:32},(_,i)=>i):[lane,lane+8,lane+16,lane+24];
 const shown=data.nodes.filter(n=>['F','B','PP'].includes(n.kind)&&n.members.every(r=>ranks.includes(r))&&(many||n.mb===+mb));const positions={};const width=many?2350:1300, height=80+ranks.length*120;
 for(const n of shown){const y=60+n.members.reduce((a,r)=>a+ranks.indexOf(r),0)/n.members.length*120;let x;
  if(n.kind==='F'||n.kind==='B')x=many?150+seq(n.stage).indexOf(n.kind+n.mb)*130:n.kind==='F'?160:720;
  else x=many?230+seq(n.stage+(n.phase==='B'?1:0)).indexOf(n.phase+n.mb)*130:n.phase==='F'?410:1050;
  positions[n.id]={x,y};}
 let svg='<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0 0 L7 3 L0 6" fill="#788f9e"/></marker></defs>';
 for(const r of ranks){const y=60+ranks.indexOf(r)*120;svg+=`<line x1="15" x2="${width-15}" y1="${y}" y2="${y}" stroke="#e7eef2"/><text x="16" y="${y-38}" font-size="14" fill="#577184">stage ${Math.floor(r/8)} / GPU ${r}</text>`;}
 // Compress only zero-cost helper nodes, never skip hidden F/B or message nodes.
 const links=[];for(const n of shown){const visit=(id,seen)=>{if(seen.has(id))return;seen.add(id);if(positions[id]){links.push([id,n.id]);return;}const p=by[id];if(p&&['PP_READY','PP_DONE','FB_END'].includes(p.kind))p.predecessors.forEach(k=>visit(k,seen));};n.predecessors.forEach(k=>visit(k,new Set()));}
 const seen=new Set();for(const [a,b] of links){const k=a+'>'+b;if(seen.has(k))continue;seen.add(k);if(many&&!$('allEdges').checked&&selected!==a&&selected!==b&&by[a].members[0]!==by[b].members[0])continue;
  const u=positions[a],v=positions[b],forward=v.x>=u.x;const x1=u.x+(forward?58:-58),x2=v.x+(forward?-61:61);const active=selected===a||selected===b;
  svg+=`<path d="M${x1} ${u.y} C${x1+(forward?30:-30)} ${u.y},${x2+(forward?-30:30)} ${v.y},${x2} ${v.y}" fill="none" stroke="${active?'#bc4d46':'#8fa5b3'}" stroke-width="${active?2.5:1.3}" opacity="${active?1:.65}" marker-end="url(#arrow)"/>`;}
 for(const n of shown){const {x,y}=positions[n.id],p=data.parameters[n.parameter],label=n.kind==='PP'?(n.phase==='F'?'激活':'梯度')+n.mb+' '+n.members.join('↔'):n.kind+n.mb;
  svg+=`<g class="node" tabindex="0" role="button" data-id="${n.id}" aria-label="${escape(n.id)}"><rect x="${x-58}" y="${y-26}" width="116" height="52" rx="5" fill="${color[n.kind]}" stroke="${selected===n.id?'#b87920':'#8ca5b5'}" stroke-width="${selected===n.id?3:1}"/><text x="${x}" y="${y-3}" text-anchor="middle" font-size="14">${label}</text><text x="${x}" y="${y+16}" text-anchor="middle" font-size="11" fill="#577184">${escape(n.parameter)} · 试点</text></g>`;}
 $('graph').setAttribute('viewBox',`0 0 ${width} ${height}`);$('graph').style.minWidth=width+'px';$('graph').innerHTML=svg;
 document.querySelectorAll('.node').forEach(el=>{const act=()=>{details(by[el.dataset.id]);draw();};el.onclick=act;el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();act();}};});
 $('viewNote').textContent=(all?'显示全部32卡':'显示PP链 '+ranks.join(' → '))+'；'+(many?'显示8个微批。为避免箭头拥挤，默认突出本地及选中节点的依赖。':'仅显示微批'+mb+'，其余微批依赖没有删除，只在此视图隐藏。')+' 当前'+shown.length+'个F/B或通信节点。';
 document.querySelectorAll('[data-rank]').forEach(b=>b.classList.toggle('active',+b.dataset.rank%8===lane));window.visibleNodes=shown.length;
}
async function init(){const response=await fetch('graph.json');if(!response.ok)throw Error('读取图失败');data=await response.json();by=Object.fromEntries(data.nodes.map(n=>[n.id,n]));window.model=data;
 $('chain').innerHTML=Array.from({length:8},(_,i)=>`<option value="${i}">${[i,i+8,i+16,i+24].join(' → ')}</option>`).join('');$('micro').innerHTML=Array.from({length:8},(_,i)=>`<option value="${i}">${i}</option>`).join('')+'<option value="all">全部8微批</option>';
 $('stages').innerHTML=Array.from({length:4},(_,s)=>`<div class="stage"><h3>PP stage ${s} · ${s===0||s===3?2:4}层</h3><p>${s===0?'首部':s===3?'尾部':'中间模板'} · 8 rank</p><div class="rank-list">${data.topology.filter(t=>t.stage===s).map(t=>`<button data-rank="${t.rank}">${t.rank}</button>`).join('')}</div></div>`).join('');
 document.querySelectorAll('[data-rank]').forEach(b=>b.onclick=()=>{$('chain').value=+b.dataset.rank%8;selected=null;draw();});
 $('tail').innerHTML=Array.from({length:4},(_,s)=>`<div class="tail-row"><b>Stage ${s}<br><small>rank ${s*8}–${s*8+7}</small></b><span>最后B</span><span class="pending">⇢</span>${['DP_RS','OPT','DP_AG'].map((k,i)=>`${i?'<span class="pending">⇢</span>':''}<button class="tail-box" data-tail="s${s}:${k}">${k}<br><small>耗时与依赖待确认</small></button>`).join('')}</div><details><summary>查看stage ${s} 实测调用证据（位置不等于GPU依赖）</summary><pre>${escape(JSON.stringify(data.tail_facts.filter(x=>x.stage===s),null,2))}</pre></details>`).join('');
 document.querySelectorAll('[data-tail]').forEach(b=>b.onclick=()=>{details(by[b.dataset.tail]);draw();});
 $('params').innerHTML=Object.entries(data.parameters).map(([k,p])=>`<tr><td>${k}</td><td>${p.value_ms===null?'待测':p.value_ms.toFixed(3)}</td><td>${p.value_ms===null?'待验证的共享接口':'旧4-rank试点值，未重拟合'}</td></tr>`).join('');
 const count=Object.keys(data.parameters).length,missing=Object.values(data.parameters).filter(x=>x.value_ms===null).length;
 $('paramNote').textContent=`${count}个候选共享槽位：6个F/B、2个PP、${missing}个更新尾段。不是已证明的最优参数数量；数值缺失不填零。`;
 $('counts').textContent=`完整机读图：${data.nodes.length}节点、${data.edges.length}条边，其中512个F/B、384个PP消息。尾段候选边用状态字段单独标记。`;
 for(const id of ['chain','micro','allRanks','allEdges'])$(id).onchange=()=>{selected=null;draw();};$('reset').onclick=()=>{$('chain').value='0';$('micro').value='0';$('allRanks').checked=false;$('allEdges').checked=false;selected=null;draw();};draw();
}
init().catch(e=>{$('inspect').textContent='加载失败：'+e.message;window.loadError=e.message;});
