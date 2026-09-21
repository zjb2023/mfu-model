// Evaluator-only panel; no model costs are changed.
(()=>{
const d=DIAG,section=document.createElement('section');section.id='overestimate';section.className='panel';
const put=(parent,tag,text,cls)=>{const n=document.createElement(tag);n.textContent=text;if(cls)n.className=cls;parent.append(n);return n;};
put(section,'h2','1F1B高估在哪里？主要是中间stage的B');
put(section,'p','保持原DAG和PP成本不变，用目标实测块时长做诊断替换。仅替换中间B，误差由+6.16%降为+0.75%；这不是新模型的预测精度。','warning');
const names={middle_F:'中间F',middle_B:'中间B',first:'首stage F/B',last:'尾stage F/B'};
put(section,'h3','对1.257秒高估的贡献 · 按依赖重新调度，不直接加总块误差');
Object.entries(d.shapley_reduction_ms).concat([['residual',d.residual_after_all_fb_ms]]).forEach(([key,value])=>{
 const row=document.createElement('div');row.style.cssText='display:grid;grid-template-columns:140px minmax(0,1fr) 110px;gap:8px;align-items:center;margin:10px 0';
 put(row,'span',names[key]||'未闭合残差');const track=document.createElement('div');track.style.cssText='background:#edf0f3;height:24px';const bar=document.createElement('div');bar.style.cssText='height:24px;width:'+Math.abs(value)/12+'%;background:'+(value<0?'#6a7988':key==='middle_B'?'#8c4873':'#315d98');track.append(bar);row.append(track);put(row,'span',(value>=0?'+':'')+value.toFixed(1)+' ms');section.append(row);
});
put(section,'p','正值：该组源成本带来的高估贡献。负残差：全部F/B换成目标实测后，DAG仍比实测短141.5ms。残差可能含PP、连接/同步、包络及聚合近似，不能直接命名为纯PP误差。四组按24种替换顺序求平均，贡献与残差合计等于原偏差。','small');
const table=(parent,heads,rows)=>{const wrap=document.createElement('div');wrap.className='scroll';const t=document.createElement('table'),h=document.createElement('tr');heads.forEach(x=>put(h,'th',x));const head=document.createElement('thead');head.append(h);t.append(head);const body=document.createElement('tbody');rows.forEach(xs=>{const tr=document.createElement('tr');xs.forEach(x=>put(tr,'td',x));body.append(tr);});t.append(body);wrap.append(t);parent.append(wrap);};
table(section,['诊断条件','1F1B时间 s','相对误差'],[['原预测','baseline'],['只换中间F','middle_F'],['只换中间B','middle_B'],['换中间F+B','middle_F+middle_B'],['全部F/B','middle_F+middle_B+first+last']].map(([label,k])=>{const x=d.scenarios[k];return[label,(x.window_ms/1000).toFixed(3),(x.error_percent>0?'+':'')+x.error_percent.toFixed(2)+'%'];}));
const details=document.createElement('details');put(details,'summary','逐stage F/B对照：每项为4个microbatch均值（点击展开）');
table(details,['PP / 代表rank','方向','预测 ms','实测均值 ms','预测−实测 ms'],d.stage_stats.map(x=>['PP'+x.stage+' / rank'+x.rank,x.phase,x.predicted_ms.toFixed(3),x.observed_mean_ms.toFixed(3),x.error_mean_ms.toFixed(3)]));section.append(details);
put(section,'p','范围：256卡iter60，每stage一个代表rank，共16rank、128块；不是全256rank。使用既有边界缓存，本轮没有重扫原始trace。块时长用rank本地时间差，不拟合跨机时钟。局部块间gap包含其他stage执行与等待，不能逐stage相加。','small');
put(section,'p','下一步：核对32卡PP1与PP2的B成本，并拆查B内重计算、专家反向、CP/EP和保留间隔，先定位为什么源B更长；暂不以整体k掩盖差异。');
const link=put(section,'a','诊断报告与16组消融数据');link.href='diagnostic-report.json';
document.getElementById('pipeline').before(section);const navlink=put(document.querySelector('nav'),'a','1F1B误差归因');navlink.href='#overestimate';
})();
