"""One collection matrix for every current 16-GPU candidate."""
from html import escape


def render(rows):
    rendered = []
    for r in rows:
        strategy = '/'.join(str(r[k]) for k in ['pp', 'cp', 'tp', 'dp', 'ep', 'expert_tp'])
        rendered.append('<tr><th>'+escape(r['experiment_id'])+'</th><td>'+r['priority']+'</td><td>'+strategy+'</td><td>'+r['micro_batch_size']+' / '+r['microbatches']+' / '+r['global_batch_size']+'</td><td class="required">必采：全16rank<br>2–3个完整迭代</td><td>必须</td><td class="off">不采</td><td class="off">不采</td><td class="off">不新增</td></tr>')
    return '''<div id="sixteen-charts"><style>#s16-telemetry{min-width:1050px}#s16-telemetry .required{background:#d9efeb;color:#075d60}#s16-telemetry .off{background:#f0f3f6;color:#556570}#s16-telemetry th{white-space:nowrap}</style>
<h2>图三 · 16卡实验与采集安排</h2><p>原10组与4个并行策略合并为14组；核心6组、扩展8组。图一与图二已移除。PP/CP/TP/DP/EP/ETP分别表示流水、上下文、张量、数据、专家并行及专家张量切分。</p>
<div class="scroll"><table id="s16-telemetry"><thead><tr><th>实验</th><th>优先级</th><th>PP/CP/TP/DP/EP/ETP</th><th>MBS / MB / GBS</th><th>Profiler trace</th><th>配置/训练日志<br>已有显存峰值</th><th>DeepEP日志</th><th>MTLINK</th><th>NIC</th></tr></thead><tbody>'''+''.join(rendered)+'''</tbody></table></div>
<p class="note">每个实际执行配置均采完整F/B、重计算、梯度同步与optimizer。另保留无profiler的稳定迭代时间。DeepEP通信库照常运行，仅关闭额外日志；MTLINK不采，NIC不新增。07留出配置的trace只供封存后的评分使用。</p><p class="muted">这是待执行计划，不是已测结果。后端兼容性、并行分组和70 GB显存准入仍需逐组检查。</p></div>'''
