(()=>{
const box=document.getElementById('b-evidence'),heading=box.querySelectorAll('h3')[1],d=B_EVIDENCE;
heading.textContent='B块时长偏差解释了多少1F1B误差？';
while(heading.nextSibling)heading.nextSibling.remove();
const original=d.baseline_ms-d.truth_ms,mid=d.baseline_ms-d.middle_b_ms,all=d.baseline_ms-d.all_b_ms;
function p(text,cls){const n=document.createElement('p');n.textContent=text;if(cls)n.className=cls;box.append(n);}
p('256卡iter60原本高估 '+(original/1000).toFixed(3)+' 秒（+6.16%）。其中，中间PP1–PP14的B块时长偏差可解释 '+(mid/1000).toFixed(3)+' 秒，即原高估量的 '+(100*mid/original).toFixed(1)+'%；剩余 '+(d.middle_b_ms-d.truth_ms).toFixed(1)+' ms，对1F1B实测时间的误差为 +'+d.middle_b_error_percent.toFixed(2)+'%。','warning');
p('加入首尾stage的B后，全部B合计可解释 '+(all/1000).toFixed(3)+' 秒，占原高估量的 '+(100*all/original).toFixed(1)+'%。首尾B额外解释 '+(all-mid).toFixed(1)+' ms；还剩 '+(d.all_b_ms-d.truth_ms).toFixed(1)+' ms 未被B替换消除，对1F1B实测时间的误差为 +'+d.all_b_error_percent.toFixed(2)+'%。');
p('因此，当前1F1B高估主要来自B成本迁移偏差。这里的“B偏差”同时包含统一源模板与各stage、各microbatch实测时长之间的差异；尚未把整体偏移与位置波动单独分离，不能说位置波动本身解释了上述全部比例。');
p('计算方法：保持F、PP通信成本和依赖不变，将B块时长换成目标实测后重新调度，取总偏差减少量。解释比例的分母是原高估1.257秒；剩余0.75%/0.53%的分母是实测1F1B的20.396秒。这是iter60的误差归因，不是改进后的独立预测精度，也未代表iter40/80。','small');
})();
