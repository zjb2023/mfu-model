(()=>{
 const $=id=>document.getElementById(id),col={F:'#287eb4',B:'#cd8438',EP:'#8966b4',DP:'#008b91',OPT:'#398252'};
 function build(n0){const ns=[],by={},counts=[n0,8-n0];function add(id,name,kind,lane,duration,pred){const n={id,name,kind,lane,duration,pred};n.start=Math.max(0,...pred.map(p=>by[p].end));n.end=n.start+duration;ns.push(n);by[id]=n;return id;}
 const att=[0,1].map(r=>add('att'+r,`GPU${r} Attention前向（本地4个token）`,'F',r*2,2,[]));
 const dispatch=add('dispatch','前向分发：原卡token → 专家所在卡','EP',1,1,att);
 const ef=[0,1].map(r=>add('expertF'+r,`GPU${r} 专家前向：${counts[r]}个token`,'F',r*2,counts[r],[dispatch]));
 const combine=add('combine','前向汇总：专家输出 → token原卡','EP',1,1,ef);
 const loss=[0,1].map(r=>add('loss'+r,`GPU${r} 损失及初始梯度`,'B',r*2,1,[combine]));
 const gradSend=add('grad-send','反向交换：输出梯度 → 专家所在卡','EP',1,1,loss);
 const eb=[0,1].map(r=>add('expertB'+r,`GPU${r} 专家反向：${counts[r]}个token`,'B',r*2,2*counts[r],[gradSend,ef[r]]));
 const gradReturn=add('grad-return','反向交换：输入梯度 → token原卡','EP',1,1,eb);
 const ab=[0,1].map(r=>add('attB'+r,`GPU${r} Attention反向`,'B',r*2,3,[gradReturn,att[r]]));
 const dp=add('shared-grad','共享参数（Attention等）梯度同步','DP',3,1,ab);
 [0,1].forEach(r=>add('opt'+r,`GPU${r} 更新共享参数及本地专家参数`,'OPT',r*2,1,[dp,eb[r]]));
 return{parameters:{world_size:2,pp:1,tp:1,ep:2,shared_parameter_dp:2,top_k:1,tokens_per_origin:4,expert_tokens:counts,units:'illustrative ms',assumptions:'whole-batch synchronous EP exchanges, no overlap, fixed communication cost'},nodes:ns,by,edges:ns.flatMap(n=>n.pred.map(p=>({src:p,dst:n.id}))),total:Math.max(...ns.map(n=>n.end))};}
 const base=build(4);let current,selected='combine';
 function draw(){const n0=+$('ep-tokens').value;current=build(n0);window.epDemoState=current;const counts=[n0,8-n0];$('ep-token-value').textContent=counts.join(' / ');
 const a=Math.ceil(n0/2),b=n0-a;$('ep-routing').textContent=`GPU0原始4个token：${a}个去GPU0专家、${4-a}个去GPU1专家；GPU1原始4个token：${b}个去GPU0专家、${4-b}个去GPU1专家。`;
 const wait=Math.abs(n0-(8-n0));$('ep-summary').textContent=`专家负载 ${counts.join(' + ')} 个token：前向较快专家等待 ${wait} ms，反向较快专家等待 ${wait*2} ms。整轮从均衡基准 ${base.total} ms → ${current.total} ms（增加 ${current.total-base.total} ms）。固定总有效工作量时，当前MFU / 均衡MFU = ${base.total}/${current.total} = ${(base.total/current.total).toFixed(4)}。等待是依赖算出的时间差，不再次加入通信成本。`;
 const scale=Math.max(base.total,current.total)+1,x=t=>180+t/scale*1180,y=l=>65+l*80,mode=$('ep-edges').value;
 let svg='<defs><marker id="ep-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 Z" fill="#607d94"/></marker></defs><rect width="1400" height="400" fill="white"/>';
 ['GPU0 · 专家组0','EP分发 / 汇总','GPU1 · 专家组1','共享参数梯度同步'].forEach((name,i)=>svg+=`<text x="6" y="${y(i)+22}" font-size="14">${name}</text>`);
 for(let t=0;t<=scale;t+=2)svg+=`<text x="${x(t)}" y="24" font-size="12">${t} ms</text><line x1="${x(t)}" x2="${x(t)}" y1="35" y2="365" stroke="#e3ebf0"/>`;
 for(const e of current.edges){const focus=e.src===selected||e.dst===selected;if(mode==='none'||mode==='local'&&!focus)continue;const p=current.by[e.src],n=current.by[e.dst],sx=x(p.end)-2,tx=x(n.start)+2,down=n.lane>p.lane,sy=y(p.lane)+(down?34:0),ty=y(n.lane)+(down?0:34);let d=`M${sx} ${sy} C${sx} ${(sy+ty)/2} ${tx} ${(sy+ty)/2} ${tx} ${ty}`;if(n.lane===p.lane)d=`M${sx} ${y(p.lane)} C${sx} ${y(p.lane)-20} ${tx} ${y(n.lane)-20} ${tx} ${y(n.lane)}`;svg+=`<path data-ep-edge="${e.src}>${e.dst}" d="${d}" stroke="${focus?'#123f70':'#8aa0b1'}" stroke-width="${focus?2.4:1}" opacity="${focus?1:.5}" fill="none" marker-end="url(#ep-arrow)" pointer-events="none"/>`;}
 const labels={att0:'Attention F',att1:'Attention F',dispatch:'分发',combine:'汇总','grad-send':'梯度分发','grad-return':'梯度返回','shared-grad':'共享梯度',loss0:'损失',loss1:'损失',opt0:'更新',opt1:'更新'};
 for(const n of current.nodes)svg+=`<g data-ep-node="${n.id}" tabindex="0" role="button" aria-label="${n.name}"><title>${n.name} ${n.start}→${n.end} ms</title><rect x="${x(n.start)}" y="${y(n.lane)}" width="${x(n.end)-x(n.start)}" height="34" fill="${col[n.kind]}" stroke="${n.id===selected?'#123f70':n.end>base.by[n.id].end?'#da3434':'white'}" stroke-width="${n.id===selected?3:2}"/><text x="${x(n.start)+3}" y="${y(n.lane)+21}" font-size="10" fill="white">${labels[n.id]|| (n.id.startsWith('expert')?'专家 '+n.kind:'Attention B')}</text></g>`;
 $('ep-chart').innerHTML=svg;document.querySelectorAll('[data-ep-node]').forEach(el=>{el.onclick=()=>{selected=el.dataset.epNode;draw()};el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();el.onclick()}}});
 const n=current.by[selected];$('ep-inspect').innerHTML=`<b>${n.name}</b><p>开始 ${n.start} ms + 自身耗时 ${n.duration} ms = 结束 ${n.end} ms。</p><p>前驱：${n.pred.map(id=>current.by[id].name+'（'+current.by[id].end+' ms完成）').join('；')||'无，从0开始'}。${n.pred.length?'取前驱完成时间的最大值，才可以开始。':''}</p>`;
 }
 $('ep-tokens').oninput=draw;$('ep-edges').onchange=draw;for(const[id,n]of[['ep-balanced',4],['ep-skew',6]])$(id).onclick=()=>{$('ep-tokens').value=n;draw()};$('ep-export').onclick=()=>{const u=URL.createObjectURL(new Blob([JSON.stringify(current,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download='2GPU_EP2_TP1_DAG.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};draw();
})();
