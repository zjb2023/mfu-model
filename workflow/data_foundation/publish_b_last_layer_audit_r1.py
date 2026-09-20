"""Append measured audit to a new local report version, retaining earlier pages."""
import html,json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'results/data-foundation'
src=BASE/'b-last-layer-variation-r1';ui=BASE/'2111-blocks-ui-r1/df-v001'
out=ui/'b-r10-60to70-r3';out.mkdir(exist_ok=False)
r=json.loads((src/'report.json').read_text());m=r['means']
coupling=[]
for it in [60,70]:
    xs=json.loads((src/f'evidence-{it}.json').read_text())
    origin=min(x['origin_absolute_ms'] for x in xs)
    latest=max(x['origin_absolute_ms']+x['prepare_end_ms'] for x in xs)
    first=min(x['origin_absolute_ms']+x['combine_start_ms'] for x in xs)
    coupling.append(dict(iteration=it,total_expert_rows=sum(x['total_tokens'] for x in xs),latest_rank=max(xs,key=lambda x:x['origin_absolute_ms']+x['prepare_end_ms'])['rank'],latest_ready_ms=latest-origin,first_visible_combine_ms=first-origin,effective_residual_ms=first-latest))
def f(v):return f'{v:+.3f}'
def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
tokens=table(['rank','token60','token70','token变化%','FC2变化ms','FC1变化ms'],[[x['rank'],x['tokens60'],x['tokens70'],f(x['token_change_percent']),f(x['FC2_change_ms']),f(x['FC1_change_ms'])] for x in r['rows']])
combine=table(['rank','Combine包络变化ms','活动并集变化ms','未覆盖变化ms','通信流可见尾部变化ms'],[[x['rank'],f(x['combine_envelope_change_ms']),f(x['combine_union_change_ms']),f(x['combine_uncovered_change_ms']),f(x['visible_comm_tail_change_ms'])] for x in r['rows']])
parts=table(['重计算互斥时间分类','70−60 八卡平均 ms'],[[k,f(v)] for k,v in r['recompute_disjoint_delta_ms'].items()])
joins=table(['迭代','最晚准备rank','最晚准备ms','最早可见Combine后处理ms','两者间隔ms'],[[x['iteration'],x['latest_rank'],f(x['latest_ready_ms']),f(x['first_visible_combine_ms']),f(x['effective_residual_ms'])] for x in coupling])
audit=f'''<section id="last-layer-audit"><h2>新增证据 · 最后执行的第1层，为什么变慢？</h2>
<p>只诊断 iter60/70 的实际差异，未调整 DAG 或参数。token 来自重计算前向 GroupedLinear 的 tokens_per_expert，已通过序列号对应到本层反向 FC2/FC1。</p>
<h3>专家工作量与梯度计算</h3>{tokens}<p>各卡有20个本地专家。完整分布、CV和CPU/GPU事件索引见下方证据；总token不是专家负载分布的全部，不能仅按总数认定线性性能关系。</p>
<h3>Combine：区分活动和包络</h3><p>八卡平均包络变化 {f(m['combine_envelope_change_ms'])} ms，其中活动并集变化 {f(m['combine_union_change_ms'])} ms、未覆盖部分变化 {f(m['combine_uncovered_change_ms'])} ms。通信流可见尾部平均变化 {f(m['visible_comm_tail_change_ms'])} ms。</p>{combine}
<p>“未覆盖”只表示选定 Combine 关联设备活动没有覆盖这段时间，不是已证实 GPU 全空闲、等待其他卡或纯传输。可见尾部不等于全部网络服务。</p>
<h3>八卡就绪关系：最晚准备卡改变了</h3>{joins}<p>每个迭代以本层八卡最早设备开始为0。最晚准备卡由rank8变为rank11；组级有效残余仅由1.254变为1.396ms。该结果与负载变化引起就绪不均衡相符，但尚非物理barrier或纯等待的因果证明。跨卡时钟可比性沿用既有假设。</p>
<h3>重计算：不重复计费的时间分解</h3>{parts}<p>R_FC1/FC2是重计算前向；R_dispatch/combine是前向EP；R_CP是CP设备活动；R_other是其他关联活动；overlap是不同分类同时活动；uncovered是没有关联设备活动覆盖。各项互斥，可加总为重计算窗口增量。</p>
<h3>这些证据应进入什么参数？</h3><p>FC1/FC2：每专家token分布与张量尺寸 → 计算成本候选。Combine：本地准备、跨卡就绪与通信流可见后处理分开核对，不把完整包络替换成网络service。重计算：先按此互斥分解定位变化，再决定是否替换保留包络。目标token仅作事后诊断，不成为60单独预测70的隐含输入。</p>
<p><a href="audit-report.json">完整审计报告</a> · <a href="audit-manifest.json">审计输入与代码哈希</a></p></section>'''
previous=ui/'b-r10-60to70-r2'
for p in previous.iterdir():
    if p.is_file():shutil.copy2(p,out/p.name)
page=(previous/'index.html').read_text();needle='<section><h2>该卡内部组件：两次实测的差异</h2>';assert needle in page
(out/'index.html').write_text(page.replace(needle,audit+needle))
for name in ['report','manifest']:shutil.copy2(src/(name+'.json'),out/('audit-'+name+'.json'))
(out/'coupling-summary.json').write_text(json.dumps(coupling,indent=2)+'\n')
def rec(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(out/'ui-manifest.json').write_text(json.dumps(dict(version='b-r10-60to70-ui3-audit',inputs=[rec(previous/'index.html'),rec(src/'report.json'),rec(src/'manifest.json'),rec(src/'evidence-60.json'),rec(src/'evidence-70.json'),rec(Path(__file__))],outputs=[rec(p) for p in out.iterdir() if p.name!='ui-manifest.json']),indent=2)+'\n')
print(out)
