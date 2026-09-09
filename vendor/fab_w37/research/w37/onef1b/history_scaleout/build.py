"""Import reviewed Task B documentation without rerunning models or changing frozen inputs."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import html
import json
import re
import statistics
from mfu_results import section as observed_mfu_section
from mlperf_recipes import section as mlperf_recipe_section
from time_mfu_figure import replace_figure as add_time_mfu_figure
from inline_reading import integrate as inline_reading

ROOT = Path(__file__).resolve().parents[4]
DOC = ROOT / 'docs/w37/1f1b/history_scaleout_20260908'
OUT = ROOT / 'results/w37/A/history-scaleout-20260908'
PARENT = ROOT / 'results/w37/A/integration-20260907/integrated_report.html'
PREFIX = '/docs/w37/1f1b/history_scaleout_20260908/'
sha = lambda b: hashlib.sha256(b).hexdigest()
raw = (DOC / 'source8038.original.html').read_bytes()
manifest = json.loads((DOC / 'source_manifest.json').read_text())
assert sha(raw) == manifest['source_html_sha256']
for f in manifest['files']:
    assert sha((DOC / f['local_path']).read_bytes()) == f['sha256']
source = raw.decode()
match = re.search(r'<script id="report-data"[^>]*>(.*?)</script>', source, re.S)
data = json.loads(match[1])
assert len(data['points']) == 168 and len(data['labels']) == 6
for row in data['summary']:
    points = [p for p in data['points'] if p['method'] == row['method'] and p['case_id'] == row['case_id']]
    assert len(points) == row['n']
    for field, calculate in {
        'training_mape_pct': lambda p: abs(float(p['predicted_training_step_ms']) / float(p['actual_training_step_ms']) - 1) * 100,
        'mfu_relative_mape_pct': lambda p: abs(float(p['predicted_mfu_pct']) / float(p['actual_mfu_pct']) - 1) * 100,
        'mfu_mae_percentage_points': lambda p: abs(float(p['predicted_mfu_pct']) - float(p['actual_mfu_pct'])),
    }.items():
        assert abs(statistics.mean(calculate(p) for p in points) - row[field]) < 1e-7, (row['method'], field)
(DOC / 'all_method_points.csv').write_text(data['csv'])
# Same-origin snapshot keeps all original charts and CSV downloads operational.
source = source.replace('http://192.168.0.49:8037/w37-report.html#baseline', '/w37-report-history.html#baseline')
source = source.replace('<body>', '<body><aside style="padding:12px 24px;background:#edf7f4;color:#174b51;font:14px/1.6 system-ui">8038 原研究页面的本地快照 · 方法与数据已封存；下方复现命令对应原 Task B 工作区。<a target="_blank" rel="noopener" href="http://192.168.0.49:8038/w37-16to256.html">打开来源页面</a></aside>', 1)
(DOC / 'methods.html').write_text(source)

class Bounds(HTMLParser):
    def __init__(self, text):
        super().__init__(); self.offsets = [0]; self.depth = 0; self.begin = None; self.end = None
        for m in re.finditer('\n', text): self.offsets.append(m.end())
        self.feed(text)
    def position(self):
        line, col = self.getpos(); return self.offsets[line - 1] + col
    def handle_starttag(self, tag, attrs):
        if tag == 'div':
            if dict(attrs).get('id') == 'scaleout':
                assert self.begin is None
                self.begin = self.position(); self.depth = 1
            elif self.begin is not None and self.end is None: self.depth += 1
    def handle_endtag(self, tag):
        if tag == 'div' and self.begin is not None and self.end is None:
            self.depth -= 1
            if self.depth == 0: self.end = self.position() + len('</div>')

old = PARENT.read_text(); bounds = Bounds(old)
assert bounds.begin is not None and bounds.end is not None
table = []
for method, name in data['labels'].items():
    r = next(x for x in data['summary'] if x['case_id'] == 'target256' and x['method'] == method)
    table.append('<tr><td>' + html.escape(name) + '</td>' + ''.join('<td>'+f'{value:.4f}'+'</td>' for value in [r['predicted_training_step_ms']/1000, r['training_mape_pct'], r['predicted_mfu_pct'], r['mfu_mae_percentage_points'], r['mfu_relative_mape_pct']]) + '</tr>')
new = '''<div class="subsection" id="scaleout"><h3>16→256 外推：模型、三阶段对照与六种方法评价</h3>
<p>更新自8038的2026-09-08研究页面，替换此前仅包含四种方法的说明。主模型M2不变，新增独立封存的M3候选和T3三阶段模型。这里核对并整合已生成的结果，不重新执行外推模型。</p>
<div class="note"><b>边界不变：</b>16卡源校准12点、源留出8点；256卡目标20点。已有目标属于历史开发评价。源留出约3.28%不是跨规模误差，新增三阶段时间MAPE约44.81%也不证明三阶段普遍优于DAG。</div>
<div class="scroll"><table><thead><tr><th>方法</th><th>Training预测秒</th><th>Training MAPE %</th><th>预测MFU %</th><th>MFU MAE（百分点）</th><th>MFU相对MAPE %</th></tr></thead><tbody>'''+''.join(table)+'''</tbody></table></div>
<p>下方完整保留输入与校准、成本迁移、流水示意、Step→MFU分账、三阶段公式、六方法逐点对比及证据入口。168行包括6种方法各8个源留出点和20个目标点，不是168个独立目标迭代。</p>
<p><a href="'''+PREFIX+'''methods.html" target="_blank" rel="noopener">单独打开完整16→256图文报告</a> · <a href="'''+PREFIX+'''all_method_points.csv" download>下载六种方法168行结果</a> · <a href="'''+PREFIX+'''source_manifest.json">来源与文件指纹</a></p>
<iframe id="scaleout-report-frame" title="16到256卡：完整外推方法与交互结果" src="'''+PREFIX+'''methods.html" style="width:100%;height:1100px;border:1px solid #d4e1ea;border-radius:12px;background:white" loading="lazy"></iframe>
<p class="muted">框内为独立的本地完整页面，可使用其目录、方法选择、逐点图和下载按钮；单独打开可获得更大阅读区域。源证据文件也已保存到8037，无需8038在线。</p></div>'''
updated = old[:bounds.begin] + new + old[bounds.end:]
# Remove the old four-method table's dataset; its DOM was deliberately replaced.
m = re.search(r'(<script[^>]*id="report-data"[^>]*>)(.*?)(</script>)', updated, re.S)
dataset = json.loads(m[2]); assert 'scaleout-iterations' in dataset
del dataset['scaleout-iterations']
updated = updated[:m.start(2)] + json.dumps(dataset, ensure_ascii=False).replace('<', '\\u003c') + updated[m.end(2):]
updated = updated.replace('独立案例：16→256外推验证', '独立案例：16→256外推（六种方法与完整报告）')
# Presentation-only index addition; the frozen report and model outputs stay intact.
heading = '<section id="why-dag"><h2>02 · 如何预测训练时间与 MFU</h2>'
assert updated.count(heading) == 1
entry = '<div class="note" id="four-gpu-demo-index"><b>先看一个直观的小例子：</b><a href="/docs/w37/1f1b/DAG_4GPU_DEMO.html">4 卡交互图：三阶段模型与 DAG 如何预测训练时间</a>。可以调整计算和通信耗时，点选节点查看依赖与等待。实际历史结果：224 卡同九轮，三阶段 v5.4 整轮 Profiler 时间 MAPE 16.823672%，计算图 v682 为 10.464727%；<a href="#baseline">查看评价口径与证据</a>。4 卡图的耗时是假设值，与该实测评估分开。</div>'
updated = updated.replace(heading, heading + entry)
result_heading = '<section id="results"><h2>06 · 预测结果、评价口径与误差分账</h2>'
assert updated.count(result_heading) == 1
updated = updated.replace(result_heading, result_heading + observed_mfu_section(OUT))
limits_heading = '<section id="limits"><h2>08 · 外推验证、能力边界与下一步</h2>'
assert updated.count(limits_heading) == 1
updated = updated.replace(limits_heading, limits_heading + mlperf_recipe_section())
calibration_heading = '<section id="calibration"><h2>04 · 性能模型参数、校准与源到目标迁移</h2>'
assert updated.count(calibration_heading) == 1
atlas_entry = '<div id="parameter-atlas-entry" class="subsection"><p>在图中选择PP、微批和F/B，点击模块查看参数值与来源；入口、OPT、RS和AG也可放大。展示v685各PP的lane0代表卡，通信汇总行覆盖全部组；不是trace观测。</p><p><a href="/docs/w37/1f1b/PARAMETER_ITER_ATLAS.html">打开独立参数图册：整轮 → F/B模块 → 参数与来源</a> · 图册顶部可返回本章。</p></div>'
# Retain the exact historical detail and IDs, but keep chapter04's default reading path short.
body_start = updated.index(calibration_heading) + len(calibration_heading)
next_section = updated.index('<section id="development">', body_start)
body_end = updated.rfind('</section>', body_start, next_section)
assert body_end > body_start
legacy_calibration = updated[body_start:body_end]
calibration_summary = '<div id="calibration-brief"><h3>本轮校准，只记住这几项</h3><ul><li><b>计算：</b>源侧58个可用成本键，实际3键更新144个节点，位于PP0和PP13；其余计算继续继承。</li><li><b>反向PP与入口：</b>源侧960条反向消息记录映射到目标624个消息节点；入口重新拟合为1265.388210 ms。</li><li><b>其余成本：</b>前向PP、CP/EP、优化器及多数运行时成本继承。图外补差和outer单列，不塞进GPU节点。</li></ul><p>参数值不一定等于整个模块时长；节点和各卡耗时不能直接相加成整轮。预测误差与版本增量分别见第06章和第05章。</p><p><a href="#parameter-system">参数类别与来源</a> · <a href="#compute-binding-explained">144节点的位置</a> · <a href="#physical-params">全部58个计算键</a> · <a href="#pp-model-versions">前后向通信口径</a></p></div>'
updated = updated[:body_start] + atlas_entry + calibration_summary + '<details id="calibration-evidence"><summary>按需展开：完整参数表、校准方法与历史审计</summary>' + legacy_calibration + '</details>' + updated[body_end:]

updated = add_time_mfu_figure(updated, OUT)
OUT.mkdir(parents=True, exist_ok=True)
updated = inline_reading(updated, OUT)
(OUT / 'integrated_report.html').write_text(updated)
report = {'parent_sha256': sha(PARENT.read_bytes()), 'source_html_sha256': sha(raw), 'updated_sha256': sha(updated.encode()), 'scope': 'Replace scaleout subsection and its old dataset; add four-GPU teaching example index and historical metric summary to modeling chapter; add measured-clock MFU and effective compute accounting to results chapter; register MLPerf DeepSeek-V3 recipe validation TODO with pinned SimAI recipe snapshot and checked arithmetic; make parameter atlas primary in chapter04 and collapse historical detail with original anchors preserved; extend v685 four-iteration time figure with same-window MFU; keep three reference views inline; open long teaching and parameter atlases as standalone pages with return links; model outputs unchanged', 'points': 168, 'target_points_per_method': 20, 'methods': 6, 'summary_formula_checks': 36, 'model_execution': False}
(DOC / 'integration_manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print('Built history report overlay; 6 methods, 168 points, 36 summary formula checks PASS.')
