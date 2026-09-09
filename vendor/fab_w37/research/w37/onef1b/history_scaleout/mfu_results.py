"""Present inherited MFU accounting without changing any model or calibration."""
import csv
import hashlib
import io
import json
import re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[4]

def section(out):
    spec = json.loads((ROOT/'docs/w37/1f1b/integration/inputs.json').read_text())['inputs']['version_iterations']
    source = ROOT/spec['path']
    raw = source.read_bytes()
    assert len(raw) == spec['bytes'] and hashlib.sha256(raw).hexdigest() == spec['sha256'], 'Frozen metric input changed'
    config = ROOT/'case_224gpu_pp14_cp2_a2a/config/dag_v685_source_steady_calibration_2026w36.toml'
    target = config.read_text().split('[target]')[1].split('[model]')[0]
    def value(key):
        return float(re.search(r'^'+key+r'\s*=\s*([\d.e+]+)', target, re.M)[1])
    flops, peak, n = value('model_flops_per_iteration'), value('peak_tflops_per_gpu'), int(value('world_size'))
    assert n == 224 and peak == 500 and flops == 8.436548311982576e16
    selected = [r for r in csv.DictReader(io.StringIO(raw.decode())) if r['variant']=='v685_frozen' and r['split']=='development_primary']
    assert [int(r['iteration']) for r in selected] == [85,90,95,100]
    rows = []
    for r in selected:
        training, trace = float(r['actual_training_ms'])/1000, float(r['actual_profiler_ms'])/1000
        effective = flops/training/1e15
        mfu = effective/(n*peak/1000)*100
        assert abs(mfu-float(r['actual_mfu_pct_derived'])) < 1e-10
        rows.append(dict(iteration=int(r['iteration']), actual_profiler_seconds=trace, actual_training_seconds=training,
                         training_effective_pflops=effective, training_per_gpu_tflops=effective*1000/n,
                         training_mfu_pct=mfu, profiler_effective_pflops=flops/trace/1e15,
                         profiler_interval_ratio_pct=flops/trace/(n*peak*1e12)*100))
    out.mkdir(parents=True, exist_ok=True)
    with (out/'observed_mfu_accounting.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    evidence = dict(source=spec, config_path=str(config.relative_to(ROOT)), config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                    model_flops_per_iteration=flops, peak_tflops_per_gpu=peak, world_size=n, cluster_peak_pflops=n*peak/1000,
                    interpretation='Inherited model-useful FLOPs divided by measured time; not hardware activity or directly counted executed FLOPs. Numerator not independently verified.',
                    rows=rows, mean_training_mfu_pct=mean(r['training_mfu_pct'] for r in rows),
                    mean_training_effective_pflops=mean(r['training_effective_pflops'] for r in rows))
    (out/'observed_mfu_accounting.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
    body=''.join('<tr><td>'+str(r['iteration'])+'</td>'+''.join(f'<td>{r[k]:.6f}</td>' for k in ['actual_profiler_seconds','actual_training_seconds','training_effective_pflops','training_per_gpu_tflops','training_mfu_pct','profiler_effective_pflops','profiler_interval_ratio_pct'])+'</tr>' for r in rows)
    return '''<div class="subsection" id="observed-mfu"><h3>实测耗时对应多少 MFU、多少有效算力？</h3>
<p><b>224 卡目标，85 / 90 / 95 / 100 四轮：</b>采用训练日志整轮时间计算，MFU逐轮平均为 <b>'''+f"{evidence['mean_training_mfu_pct']:.6f}%"+'''</b>，集群模型有效算力逐轮平均为 <b>'''+f"{evidence['mean_training_effective_pflops']:.6f} PFLOP/s"+'''</b>。下表为实际耗时的换算结果，与模型预测值分开。</p>
<ul><li><b>采用的峰值：</b>S5000 单卡 500 TFLOP/s，按本报告继承的 BF16 配置口径；224 卡合计 112,000 TFLOP/s = 112 PFLOP/s。这是归一化用的峰值输入，不是本轮实测峰值。</li>
<li><b>每轮模型有效计算量：</b>8.436548311982576 × 10<sup>16</sup> FLOP（84.365483 PFLOP），继承 v685 配置。该分子尚未依据当前模型配置独立复核。</li>
<li><b>有效算力 = 每轮模型有效 FLOPs ÷ 实际秒数；MFU = 有效算力 ÷ 集群峰值 × 100%。</b>单卡平均有效算力 = 集群有效算力 ÷ 224，不能代表各卡分别达到该值。</li></ul>
<p class="note"><b>“trace 实测 MFU”需要区分时间范围：</b>trace 提供 Profiler 区间时间，训练日志提供 Training 整轮时间；MFU 是结合上述计算量和峰值换算的，不是 trace 直接测得的 GPU 忙碌率。正式 MFU 评价使用 Training 整轮时间。右两列仅以 Profiler 区间时间作分母，未计入两时钟差额，数值更高；它们只用于核对口径，不能替代整轮训练 MFU。</p>
<style>#observed-mfu-table{width:100%;min-width:900px;table-layout:fixed}#observed-mfu-table th{white-space:normal;line-height:1.5}#observed-mfu-table td{font-size:13px;padding:10px 8px}</style><div class="scroll"><table id="observed-mfu-table"><thead><tr><th>迭代</th><th>Trace / Profiler 秒</th><th>日志 / Training 秒</th><th>整轮有效算力 PFLOP/s</th><th>单卡平均有效算力 TFLOP/s</th><th>整轮 MFU %</th><th>仅 Profiler 区间有效算力 PFLOP/s</th><th>仅 Profiler 区间换算比例 %</th></tr></thead><tbody>'''+body+'''</tbody></table></div>
<p>这里“有效”指模型有用计算量的吞吐率，不等于硬件实际执行的全部浮点运算量；额外重计算、通信和等待不能当作有效 FLOPs。上文的均值均为逐轮结果的算术平均，不是用平均时间再计算。历史16.82%来自60–100九轮时间误差，与本表四轮MFU不是同一指标。</p>
<p><a href="/results/w37/A/history-scaleout-20260908/observed_mfu_accounting.csv" download>下载逐轮算力与 MFU 换算表</a> · <a href="/results/w37/A/history-scaleout-20260908/observed_mfu_accounting.json">查看输入、公式口径与文件指纹</a> · <a href="/docs/w37/1f1b/post685/delivery/version_iteration_results.csv">原始封存逐轮评价表</a></p></div>'''
