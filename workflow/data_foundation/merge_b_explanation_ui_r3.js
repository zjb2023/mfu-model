(()=>{
const box=document.getElementById('b-evidence'),duplicate=box.querySelectorAll('h3')[1];
if(duplicate){while(duplicate.nextSibling)duplicate.nextSibling.remove();duplicate.remove();}
const panel=document.getElementById('overestimate');
const heading=[...panel.querySelectorAll('h3')].find(n=>n.textContent.startsWith('对1.257秒高估的贡献'));
let explanation=heading.nextElementSibling;
while(explanation&&explanation.tagName!=='P')explanation=explanation.nextElementSibling;
const d=B_EVIDENCE,total=d.baseline_ms-d.truth_ms,mid=d.baseline_ms-d.middle_b_ms,all=d.baseline_ms-d.all_b_ms;
explanation.textContent='正值表示该组成本带来的高估贡献。中间B解释 '+(mid/1000).toFixed(3)+' 秒，占原高估量的 '+(100*mid/total).toFixed(1)+'%；仅替换中间B后，1F1B剩余误差 +'+d.middle_b_error_percent.toFixed(2)+'%（'+(d.middle_b_ms-d.truth_ms).toFixed(1)+' ms）。若连首尾B也替换，全部B合计解释 '+(100*all/total).toFixed(1)+'%，剩余 +'+d.all_b_error_percent.toFixed(2)+'%（'+(d.all_b_ms-d.truth_ms).toFixed(1)+' ms）。';
const note=document.createElement('p');note.className='small';note.textContent='图中“首/尾stage F/B”是该stage的F与B合计贡献，不是仅B贡献；上面的全部B比例来自单独的B替换诊断。负残差−141.5 ms指全部F/B都替换后仍低估的部分，不能直接叫纯PP误差。四组按24种替换顺序求平均，贡献与残差合计等于原偏差。';explanation.after(note);
const scope=document.createElement('p');scope.className='small';scope.textContent='以上是256卡iter60的诊断：保持F、PP及依赖不变，以实测B成本替换后重新调度，不是删除B或直接累加块误差。B成本差异包含整体偏移、stage与microbatch差异，尚未分离位置波动的独立贡献。87.9%/91.4%的分母是原高估量，0.75%/0.53%的分母是实测1F1B时长；均非新模型独立预测精度。';note.after(scope);
})();
