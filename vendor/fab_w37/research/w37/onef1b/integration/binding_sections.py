"""Display the independently sealed binding experiment; preserve v685 tables."""
import csv, json
from common import ROOT, digest

DOC = ROOT/'docs/w37/1f1b/binding'

def load():
    manifest = json.loads((DOC/'delivery_inputs.json').read_text())
    assert manifest['status'] == 'PASS'
    for name, record in manifest['files'].items():
        assert digest(DOC/name) == record['sha256'], name
    def rows(name):
        with (DOC/'evidence'/name).open() as f: return list(csv.DictReader(f))
    return manifest, rows

def audit_section(table, details, note):
    manifest, rows = load()
    a = json.loads((DOC/'evidence/audit_summary.json').read_text())
    assert a['direct_keys'] == 3 and a['eligible_fine_nodes'] == 70192
    s = '<div id="binding-audit"><h3>新增研究：为什么58个聚合键只直接命中3个</h3>'
    s += '<p><b>多数内部节点已经使用另一种、更细的键。</b>v685按“方向｜层类型｜上下文｜执行位置｜语义槽”完全匹配；v65内部节点则带有类型前缀，并按“方向｜微批次｜全局层号｜执行位置｜语义槽”查成本。55个未直接命中的聚合键都能关联到采用后一格式的内部节点。首尾PP的3个键是保留的边界回退；不是因为只更新PP0和PP13就足以重建内部计算。</p>'
    s += '<pre>聚合键：forward|dense|step_entry_to_dense|exposed_gap|before_cp0\n细粒度节点键：compute_gap|forward|0|0|exposed_gap|before_cp0\n                         方向  MB0 全局层0  执行位置  语义槽\n源侧成本查表时去掉 compute_gap 前缀；它标识节点成本类别。</pre>'
    s += table(['统计对象','数量','含义'],[
        ['v685本轮直接更新','3键 / 144节点','保留正式v685记录；本次独立候选不改这3个边界键'],
        ['内部细粒度节点','80,064','采用全局层号和微批次的成本键'],
        ['其中原有正计算成本','71,488','不是所有结构节点都有独立计算成本'],
        ['新候选可更新','70,192','源85/90有对应观测；覆盖224 rank、PP0–PP13、MB0–MB2'],
        ['正成本但新窗口缺观测','1,296','继续继承，不用0替代未知'],
        ['原有零成本且源表缺观测','8,576','继续保留；缺记录不证明GPU没有工作']])
    s += '<p>细粒度源表有6,765条成本记录。新候选先用源85/90取中位数，再保留v67原有rank/方向/微批次倍率；该倍率从继承预测成本恢复，71,488个旧节点重建误差均小于1 ns，未使用目标实测。它仍是该256→224配置的成本迁移，不是任意张量形状的通用性能函数。</p>'
    s += note('<b>候选的时长规则：</b>与通信并行的计算节点直接更新计算时长；暴露计算节点使用 <code>新时长 = max(旧完整节点时长, 新计算成本)</code>，余量记为未拆明的运行时残差。旧完整时长只是保守下限，尚未独立证明是纯非计算开销；该规则让暴露区间只能变长或不变。分量之和守恒不等于物理计时归属已验证。', True)
    s += '<p><a href="#binding-results">查看独立候选及消融结果</a> · <a href="/docs/w37/1f1b/binding/REPORT.md">完整研究报告</a> · <a href="/docs/w37/1f1b/binding/ROUTE.md">研究路线</a> · <a href="/docs/w37/1f1b/binding/evidence/audit_all_58_key_audit.csv">全部58键的对应原因</a> · <a href="/docs/w37/1f1b/binding/HANDOFF.md">逐节点结果与复现入口</a></p></div>'
    return s

def results_section(table, details, note):
    _, rows = load()
    names = {'frozen_v685':'正式v685','fine_both':'独立候选：两类都更新','fine_exposed_only':'消融：只更新暴露计算','fine_overlap_only':'消融：只更新重叠计算'}
    metrics = rows('evaluate_metrics.csv')
    s = '<div id="binding-results"><h3>计算成本绑定候选：源侧更新，目标四轮开发评估</h3><p>以下是新增的独立研究结果。正式v685的图、成本和上述历史表保留；候选仍使用327,746节点、364,784条依赖边。四种配置在目标评分前封存，没有按目标结果选择参数或加等待边。</p>'
    s += table(['方法','1F1B MAPE %','Profiler MAPE %','Training MAPE %','MFU相对 MAPE %','MFU绝对误差均值（百分点）'],[
        [names[r['variant']]]+[f'{float(r[k]):.4f}' for k in ['onef1b_MAPE_pct','profiler_MAPE_pct','training_MAPE_pct','mfu_relative_MAPE_pct','mfu_MAE_percentage_points']] for r in metrics])
    s += '<p>两类都更新时，1F1B预测从19.1158秒变为19.2838秒，MAPE下降0.7792个百分点；Training预测从22.7581秒变为22.9300秒。此前模型低估，所以预测变长使误差减小，不能把它描述成实际训练加速。</p>'
    s += note('<b>改善与退化同时报告：</b>收尾MAPE从35.2692%升到35.7889%，入口和outer不变。1,120个优化器节点的成本均未修改，但最晚B结束推迟167.9944 ms，整图结束推迟171.8538 ms，因此两者之间的收尾区间增加3.8594 ms。依赖传播会改变区间长度，不能把这一变化说成优化器成本被调大。', True)
    s += details('源95/100：细粒度与聚合成本使用同一校准窗口比较',
        '<p>两种方法均用源85/90校准，在相同13,116条源95/100记录上比较；另125条缺少细粒度拟合，单列不清零。WMAPE = 绝对误差之和 ÷ 实际局部成本之和；这是局部成本指标，不是整轮或1F1B MAPE。</p>'+table(['源轮次','计算位置','方法','共同样本数','MAE ms','WMAPE %'],[
            [r['iteration'],'暴露计算' if r['scope']=='exposed_gap' else '重叠计算','保留层号和MB' if r['method']=='fine_global_layer_mb' else '按上下文聚合',r['rows'],f"{float(r['MAE_ms']):.4f}",f"{float(r['WMAPE_pct']):.4f}"] for r in rows('review_source_method_comparison.csv')]))
    point = rows('evaluate_iteration_results.csv')
    s += details('目标224卡：正式v685与两类更新候选的逐轮结果',table(['方法','轮次','实际1F1B秒','预测1F1B秒','1F1B APE %','实际Training秒','预测Training秒','Training APE %','MFU相对APE %'],[
        [names[r['variant']],r['iteration']]+[f'{float(r[k])/1000:.4f}' for k in ['onef1b_actual_ms','onef1b_predicted_ms']]+[f"{float(r['onef1b_APE_pct']):.4f}"]+[f'{float(r[k])/1000:.4f}' for k in ['training_actual_ms','training_predicted_ms']]+[f'{float(r[k]):.4f}' for k in ['training_APE_pct','mfu_relative_APE_pct']] for r in point if r['variant'] in ['frozen_v685','fine_both']]))
    s += '<p><b>结论：</b>细粒度源成本的局部误差与目标开发误差都有改善，支持继续研究；但旧时长下限未独立验证，兼容新键的源侧整轮回放仍缺失，收尾还有退化，因此暂不升级正式版本。源95/100已被历史继承成本使用、目标四轮也已暴露，均不属于独立盲测。</p>'
    s += '<p><a href="/docs/w37/1f1b/binding/evidence/evaluate_iteration_results.csv">所有候选逐轮Step/MFU</a> · <a href="/docs/w37/1f1b/binding/evidence/evaluate_phase_iteration_results.csv">全部阶段逐轮结果</a> · <a href="/docs/w37/1f1b/binding/evidence/review_tail_propagation.csv">收尾传播证据</a> · <a href="/docs/w37/1f1b/binding/research_acceptance.json">输入、代码、输出哈希与验收</a></p></div>'
    return s
