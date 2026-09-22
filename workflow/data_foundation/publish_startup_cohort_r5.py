"""Publish verified rank0 means; keep unavailable startup/AG values explicit."""
import hashlib,html,json
from pathlib import Path
from audit_startup_cohort_r5 import stats,save
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/data-foundation'
REL=Path('results/data-foundation/2111-blocks-ui-r1/df-v001')
MIRROR=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fmt(x):return 'N/A' if x is None else f'{x:.3f}'
def main():
    large=BASE/'startup-cohort-r5';small=BASE/'startup-cohort-r5-16-verified'
    a=json.loads((large/'report.json').read_text());b=json.loads((small/'report.json').read_text())
    rows=[r for r in a['rows'] if r['world']!=16]+b['rows'];assert len({(r['case'],r['iteration']) for r in rows})==len(rows)
    summaries=[]
    for case in ['256','224','32']+sorted({r['case'] for r in rows if r['world']==16}):
        rs=[r for r in rows if r['case']==case];assert all(r['status']!='FAIL' for r in rs)
        summaries.append(dict(case=case,world=rs[0]['world'],traces=len(rs),iterations=[r['iteration'] for r in rs],startup=stats([r['startup_GPU_ms'] for r in rs]),AG=stats([r['selected_AG_ms'] for r in rs]),all_AG_sum=stats([r['all_timer_AG_sum_ms'] for r in rs]),missing_AG_iterations=[r['iteration'] for r in rs if r['selected_AG_ms'] is None],AG_call_counts=sorted({r['timer_AG_count'] for r in rs}),exclude100_startup=stats([r['startup_GPU_ms'] for r in rs if r['iteration']!=100]),exclude100_AG=stats([r['selected_AG_ms'] for r in rs if r['iteration']!=100])))
    out=BASE/'startup-cohort-r5-summary';out.mkdir(exist_ok=False)
    save(out/'report.json',dict(summaries=summaries,rows=rows,rank_scope='rank0 cross-iteration mean, not world-rank mean',small_selection=b['selection'],missing_trace='16-8211 iteration98 absent; no fabricated sample'))
    large_table='|case|样本数|整个启动均值 ms|小AG均值 ms|其余启动均值 ms|\n|---|---:|---:|---:|---:|\n'
    for s in summaries[:3]:large_table+=f'|{s["case"]}|{s["traces"]}|{fmt(s["startup"]["mean"])}|{fmt(s["AG"]["mean"])}|{fmt(s["startup"]["mean"]-s["AG"]["mean"])}|\n'
    small_table='|16卡case|可用trace数|末次统计AG有效样本|AG均值 ms|缺关联迭代|\n|---|---:|---:|---:|---|\n'
    for s in summaries[3:]:small_table+=f'|{s["case"][3:]}|{s["traces"]}|{s["AG"]["n"]}|{fmt(s["AG"]["mean"])}|{s["missing_AG_iterations"]}|\n'
    doc='''# 启动/统计AG跨迭代均值 r5

## 口径与选择

全部为rank0的跨迭代算术均值，不是全rank均值，不是全局makespan。256/224/32卡各20份trace：目录iter5,10,…,100，启动是ProfilerStep开始到首forward_step关联GPU开始。同窗口首F前唯一default_pg、24-FP32全局AG。

16卡14case选目录iter8,13,…,103（首尾3/108不取）；8211缺iter98，仅19份。无同名forward_step标记，因此同口径启动均值N/A，不能用第一条kernel或DataLoader硬代。统计AG取每轮CPU记录的最后一条default_pg16、24-FP32 AllGather，与GPU按精确External id关联；它通常在迭代尾部。另保存所有候选，调用次数可能1或2。

部分16卡GPU事件缺External id和通信元数据；runtime/kernel correlation还出现差1，不能未经证明补配。未匹配的最后一条AG记缺失，不计0、不用前一条AG替代。有效样本不足20的均值是可观测子集均值，可能有缺失偏差。

## 三组大规模case

'''+large_table+'''
224卡AG均值较小不代表启动更快：整个启动均值比256卡更长，其余启动区间占主要部分。只比rank0 AG驻留会混淆“进入前耗时”和“进入后等待”。本轮没有对224所有rank做跨迭代到达诊断，不能直接解释成谁先谁后。

256卡约1秒AG不是仅iter60偶发；20轮均有这一量级。但iter60得到的99.1%到达不齐时间界不能直接推广为20轮平均等待占比。

## 16卡各策略

'''+small_table+'''
这些是统计AG均值，不是16卡启动耗时。不同策略、模型层数、采集/日志配置不同，禁止将三种大规模case与16卡合并拟合rank规模系数。

## 去掉iter100的敏感性

|case|样本数|启动均值 ms|AG均值 ms|
|---|---:|---:|---:|
'''
    for s in summaries[:3]:doc+=f'|{s["case"]}|19|{fmt(s["exclude100_startup"]["mean"])}|{fmt(s["exclude100_AG"]["mean"])}|\n'
    doc+='''
## 复现和证据

1. audit_startup_cohort_r5.py --out 新目录：大规模20轮与第一轮16卡盘点（其中16卡初版不可作为最终均值）。
2. verify_timer_ag_16_r5.py --out 新目录：使用已有14case清单定位真实rank0主机，并按最后CPU调用/精确GPU关联核验；它替代初版及host-supplement的16卡均值。
3. publish_startup_cohort_r5.py聚合固定输入startup-cohort-r5和startup-cohort-r5-16-verified，输出本报告；发布器是一次性写入，复现分析请使用独立输出目录。

原始trace路径、SHA256、事件索引、Profiler标签、缺失与逐轮数据均保留。目录迭代号不等于ProfilerStep标签，原标签单独保存。未训练、未拟合、未修改预测、未提交推送。大规模20轮均值PASS；16卡覆盖/关联缺项明确保留PARTIAL。
'''
    dp=ROOT/'docs/data-foundation/STARTUP_COHORT_R5.md';assert not dp.exists();dp.write_text(doc)
    def chart(field,title):
        width=850;height=250;mx=max(r[field] for r in rows if r['world']!=16)*1.1
        svg=f'<h2>{title}</h2><svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">'
        for j in range(5):
            y=210-j*45;svg+=f'<line x1="65" x2="820" y1="{y}" y2="{y}" stroke="#e0e5eb"/><text x="3" y="{y}" font-size="13">{mx*j/4:.0f}ms</text>'
        for case,color in [('256','#b64240'),('224','#3268aa'),('32','#2e8569')]:
            rs=sorted([r for r in rows if r['case']==case],key=lambda r:r['iteration']);points=' '.join(f'{65+(r["iteration"]-5)/95*755:.2f},{210-r[field]/mx*180:.2f}' for r in rs)
            svg+=f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>'
            for r in rs:svg+=f'<circle cx="{65+(r["iteration"]-5)/95*755:.2f}" cy="{210-r[field]/mx*180:.2f}" r="3" fill="{color}"><title>{case}卡 iter{r["iteration"]}: {r[field]:.3f}ms</title></circle>'
        svg+='<text x="65" y="240">iter5</text><text x="750" y="240">iter100</text></svg><p>红：256卡 · 蓝：224卡 · 绿：32卡；悬停查看逐轮值。</p>'
        return svg
    table='<table><tr><th>case</th><th>trace数</th><th>AG有效样本</th><th>AG均值 ms</th><th>启动均值 ms</th></tr>'
    for s in summaries:table+=f'<tr><td>{s["case"]}</td><td>{s["traces"]}</td><td>{s["AG"]["n"]}</td><td>{fmt(s["AG"]["mean"])}</td><td>{fmt(s["startup"]["mean"])}</td></tr>'
    table+='</table>'
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>启动与统计AG · 20轮均值</title><style>body{font:16px/1.7 system-ui;background:#f4f6f8;color:#203048}main{max-width:1080px;margin:auto;padding:20px}section{background:white;padding:20px;margin:18px 0;border-radius:12px}h1{font-size:27px}h2{font-size:21px}svg{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}.scroll{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 system-ui}a{color:#1761a0}</style><main><a href="../startup-arrival-r4/">← iter60全rank等待诊断</a><h1>启动窗口与全局统计AG：跨迭代均值</h1><p>rank0跨迭代均值，不是所有rank的平均。大规模各20轮；16卡AG在迭代尾部，启动边界不可直接同口径比较。</p><section>'''+chart('startup_GPU_ms','整个启动窗口')+chart('selected_AG_ms','其中：小消息全局AG驻留')+'''</section><section><h2>各case均值与覆盖</h2><div class="scroll">'''+table+'''</div><p>N/A不是0。16卡缺精确关联时仅统计有效子集，绝不以前一次AG替代最后一次。</p></section><section><h2>如何读这个结果</h2><p>256卡约1秒AG在20轮中持续出现；224卡AG较短，但整个启动反而更长。不能用单rank的AG驻留直接拟合rank数量系数。</p><p>16卡统计位置与大规模case不同，AG均值可单列观察，不能称为同口径启动均值。iter60的等待归因尚未逐轮验证。</p><a href="notes.md">完整方法与去掉iter100结果</a> · <a href="manifest.json">证据清单</a><details><summary>详细报告</summary><pre>'''+html.escape(doc)+'''</pre></details></section></main></html>'''
    sources=[large/'report.json',large/'manifest.json',small/'report.json',small/'manifest.json',dp]
    save(out/'manifest.json',dict(inputs={str(p):sha(p) for p in sources},code={str(Path(__file__).resolve()):sha(Path(__file__)),str(Path(__file__).with_name('audit_startup_cohort_r5.py')):sha(Path(__file__).with_name('audit_startup_cohort_r5.py'))},outputs={str(out/'report.json'):sha(out/'report.json')}))
    for base in [ROOT,MIRROR]:assert not (base/REL/'startup-cohort-r5').exists()
    for base in [ROOT,MIRROR]:
        dest=base/REL/'startup-cohort-r5';dest.mkdir();(dest/'index.html').write_text(page);(dest/'notes.md').write_text(doc);(dest/'report.json').write_bytes((out/'report.json').read_bytes())
        save(dest/'manifest.json',dict(inputs={str(p):sha(p) for p in sources+[out/'manifest.json']},code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(p):sha(p) for p in dest.iterdir()}))
    print(json.dumps([dict(case=s['case'],traces=s['traces'],n=s['AG']['n'],AG_ms=s['AG']['mean'],startup_ms=s['startup']['mean']) for s in summaries],ensure_ascii=False))
if __name__=='__main__':main()
