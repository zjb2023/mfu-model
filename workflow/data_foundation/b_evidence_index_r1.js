(()=>{
const box=document.createElement('div');box.id='b-evidence';box.style.cssText='border:1px solid #cbd5e0;padding:18px;margin:20px 0';
function put(tag,text,cls){const n=document.createElement(tag);n.textContent=text;if(cls)n.className=cls;box.append(n);return n;}
put('h3','B位置证据 · iter40 / 60 / 80（不合并microbatch）');
put('p','224卡与256卡分别按B0、B1、B2……绘制逐PP位置曲线，可切换原始GPU时长与相对PP1差值。iter100已按用户选择排除，不参与当前曲线与重复位置统计。');
put('p','224卡三轮各B在PP5→6均变短，PP11→12均回升；256卡未见三轮所有B共用的逐stage递减规律。位置波动不等同于执行过程持续加速，也尚未证明其原因。','small');
const link=put('a','打开完整逐B位置曲线与原始证据 →');link.href='../b-position-curves-r1/';link.target='_blank';link.rel='noopener';
const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent='在原总览页展开位置曲线';details.append(summary);const frame=document.createElement('iframe');frame.title='逐B位置证据 iter40 60 80';frame.style.height='950px';details.append(frame);details.addEventListener('toggle',()=>{if(details.open&&!frame.getAttribute('src'))frame.src='../b-position-curves-r1/';});box.append(details);
put('h3','如果消除B块时长误差，1F1B还剩多少误差？');
const d=B_EVIDENCE;
put('p','同一256卡iter60：仅替换中间PP1–PP14的B，得到 '+(d.middle_b_ms/1000).toFixed(3)+' 秒，对比实测 '+(d.truth_ms/1000).toFixed(3)+' 秒，剩余误差 +'+d.middle_b_error_percent.toFixed(2)+'%（约 '+(d.middle_b_ms-d.truth_ms).toFixed(1)+' ms）。','warning');
put('p','若连首尾stage也包含，即替换PP0–PP15的全部B，得到 '+(d.all_b_ms/1000).toFixed(3)+' 秒，剩余误差 '+(d.all_b_error_percent>=0?'+':'')+d.all_b_error_percent.toFixed(2)+'%（'+(d.all_b_ms-d.truth_ms).toFixed(1)+' ms）。F成本、PP通信成本与依赖关系均保持不变。');
put('p','“消除B误差”指用目标实测替换B块时长再调度，不是删除B、把B耗时设为0，也不是直接从总时间减掉所有B的误差。这是定位原因的oracle诊断；+0.75%仅属于iter60中间B替换，不能当作iter40/80或新模型的预测精度。','small');
const json=put('a','查看中间B / 全部B诊断数据');json.href='b-only-diagnosis.json';
const panel=document.getElementById('overestimate');panel.querySelector('h2').after(box);
})();
