"""Augment existing plotted page with both EDP replicas; base publisher runs first."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
dest=BASE/'2111-blocks-ui-r1/df-v001/layer-workload-r1'
data=json.loads((BASE/'layer-edp-pair-r1/report.json').read_text());assert data['status']=='PASS'
html=(dest/'index.html').read_text()
assert 'const PAIR_DATA=' not in html,'Run base publisher first; avoid duplicate augmentations'
start=html.index('<section><h2>② 完整EP8:') if '<section><h2>② 完整EP8:' in html else html.index('<section><h2>② 完整EP8：')
end=html.index('</section>',start)+len('</section>')
html=html[:start]+'''<section><h2>② 整个中间PP stage：两个EP8组分别看，再合计</h2><p><strong>固定256卡iter60，microbatch跟随上方选择。</strong>组A为各stage的lane0～7，组B为lane8～15；合计覆盖该stage全部16张卡。横轴是模型逻辑层，每4层切换stage及其物理卡组，边界已断线。仅覆盖PP1～14，不含首尾stage。</p><div class="legend"><span style="color:#00877a">● EP8组A（lane0～7）</span><span style="color:#ad4b20">● EP8组B（lane8～15）</span><span style="color:#6557a0">● 整个stage合计（仅工作量）</span></div><h3>专家实际接收token条目数：组A、组B、16卡合计</h3><div id="total"></div><div id="pair-summary"></div><h3>各EP8组最忙rank的B专家FC时间（ms）</h3><p class="muted">两组并行，耗时不相加；此处各自取组内最大FC计算成本，不等于整层DAG完成时间。</p><div id="maxcost"></div><h3>各EP8组内部工作量不均衡程度（CV）</h3><div id="cv"></div><p class="muted">本轮仅iter60的两组完整覆盖，没有证明整stage曲线跨iter稳定。上方①仍是组A lane0的三轮代表rank曲线，不要混用范围。</p><p><a href="edp-pair.md">两组核对说明</a> · <a href="edp-pair-report.json">结果</a> · <a href="edp-pair-manifest.json">来源清单</a></p></section>'''+html[end:]
start=html.index('<section id="filtering">');end=html.index('</section>',start)+len('</section>')
html=html[:start]+'''<section id="filtering"><h2>④ 两个EP8组合计：原始分配与实际进入计算</h2><p>绿色为按trace输入8192×top6×16推导的原始分配786432条；橙色为16卡实际进入专家的条目数。两个组分别完成发送—接收—FC输入核对后才相加。不是去重token数。</p><div class="legend"><span style="color:#00877a">● 原始top6分配（推导）</span><span style="color:#ad4b20">● 实际进入专家（两组实测合计）</span></div><div id="filtercurve"></div><div id="filtercomparison"></div><p>两者差额是推导的未进入计算分配量，不是直接测得的容量丢弃数。容量筛选及零概率屏蔽的独立贡献仍无法区分。每层重新路由，不是token沿层被逐渐消耗；首末减少不代表相邻层单调下降。</p><p><a href="filtering.md">第一组核对方法与边界（历史范围）</a> · <a href="edp-pair.md">两组扩展说明</a></p></section>'''+html[end:]
js='''<script>const PAIR_DATA=__DATA__;function pairDraw(){let rs=PAIR_DATA.rows.filter(r=>r.mb===+mb.value);function series(label,key,group){return {label,values:rs.map(r=>({x:r.layer,y:r[key],tip:' PP'+r.stage+' '+(group==='A'?'rank'+r.ranksA[0]+'～'+r.ranksA[7]:group==='B'?'rank'+r.ranksB[0]+'～'+r.ranksB[7]:'两组共16卡')}))}}chart('total',[series('组A','groupA','A'),series('组B','groupB','B'),series('整stage合计','total','both')]);chart('maxcost',[series('组A','max_cost_A','A'),series('组B','max_cost_B','B')]);chart('cv',[series('组A','cv_A','A'),series('组B','cv_B','B')]);chart('filtercurve',[series('原始分配（推导）','offered','both'),series('实际接收（实测）','total','both')]);let c=PAIR_DATA.comparisons.find(r=>r.mb===+mb.value),a=c.first,b=c.last;document.getElementById('pair-summary').innerHTML='<div class="scroll"><table><tr><th>当前MB'+mb.value+'</th><th>L'+a.layer+'</th><th>L'+b.layer+'</th><th>首末变化</th></tr>'+[['组A','groupA'],['组B','groupB'],['16卡合计','total']].map(([label,k])=>'<tr><td>'+label+'</td><td>'+a[k].toLocaleString()+'</td><td>'+b[k].toLocaleString()+'</td><td>'+((b[k]-a[k])/a[k]*100).toFixed(2)+'%</td></tr>').join('')+'</table></div>';document.getElementById('filtercomparison').textContent='当前MB'+mb.value+'：原始分配两端均为'+a.offered.toLocaleString()+'条；实际接收从'+a.total.toLocaleString()+'变为'+b.total.toLocaleString()+'条。推导未进入计算量从'+a.not_entered.toLocaleString()+'变为'+b.not_entered.toLocaleString()+'条。数量核对不等于容量因果归因。';}for(const id of ['world','mb','phase'])document.getElementById(id).addEventListener('change',pairDraw);pairDraw();</script>'''.replace('__DATA__',json.dumps(data,separators=(',',':')))
html=html.replace('横轴是模型逻辑层，每4层切换stage及其物理卡组，边界已断线。','横轴是模型逻辑层，折线跨PP连续连接；每4层切换stage及其物理卡组，保留PP分区标记。')
# Present the observed decrease first, then the mechanism and evidence limits.
origin=json.loads((BASE/'route-filter-origin-r1/report.json').read_text())
observed={(r['group'],r['layer']):r['active_experts'] for r in origin['stats'] if r['mb']==0}
assert observed=={('A',3):59,('A',58):29,('B',3):59,('B',58):29}
active_note='''<h3>已有结果：实际参与工作的专家也变少了</h3><p><strong>固定样本：256卡iter60、MB0，以下表格不随上方microbatch选择切换。</strong>每组始终配置160个专家，这里数的是该层实际收到至少1条token输入的专家。</p><div class="scroll"><table><tr><th>范围</th><th>L3实际收到输入</th><th>L58实际收到输入</th><th>每组配置的专家总数</th></tr><tr><td>EP8组A</td><td>59个</td><td>29个</td><td>160个，不变</td></tr><tr><td>EP8组B</td><td>59个</td><td>29个</td><td>160个，不变</td></tr></table></div><p>通俗说：每层都有160位“专家”，但这次L3有59位接到工作，L58只有29位接到工作；其余专家仍存在，只是这次没有收到输入。<strong>两个EDP副本都观察到了这个现象，不是只看其中一组造成的。</strong></p><p>这与实际进入计算的条目减少相吻合，但它是<strong>筛选后的结果</strong>：不能仅凭59→29就证明筛选前路由更集中，也不能算出容量筛选到底丢了多少。后面的④说明已有机制证据和仍缺的计数。L3、L58属于不同stage、不同模型层，并非同一批专家沿层消失；此端点对比也不代表逐层单调下降。</p>'''
html=html.replace('<div id="filtercomparison"></div>','<div id="filtercomparison"></div>'+active_note)
start=html.index('<section id="filtering">');end=html.index('</section>',start)+len('</section>')
analysis_section=html[start:end].replace('④ 两个EP8组合计','③ 两个EP8组合计')
html=html[:start]+html[end:]
anchor='<section><h2>③ 容量筛选证据与边界</h2>'
assert anchor in html
html=html.replace(anchor,analysis_section+'<section id="capacity-evidence"><h2>④ 容量筛选证据与边界</h2>',1)
html=html.replace('</html>',js+'</html>');(dest/'index.html').write_text(html)
for source,name in [(ROOT/'docs/data-foundation/LAYER_EDP_PAIR_R1.md','edp-pair.md'),(BASE/'layer-edp-pair-r1/report.json','edp-pair-report.json'),(BASE/'layer-edp-pair-r1/manifest.json','edp-pair-manifest.json')]:
    (dest/name).write_bytes(source.read_bytes())
manifest=json.loads((dest/'manifest.json').read_text())
for p in [dest/'index.html',dest/'edp-pair.md',dest/'edp-pair-report.json',dest/'edp-pair-manifest.json',BASE/'route-filter-origin-r1/report.json',Path(__file__).resolve()]:manifest['evidence'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
manifest['ui_revision']='r10: section4 presents verified findings, not discussion questions or collection plans; order, plots and counts unchanged.'
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Published both EP8 groups',dest)
