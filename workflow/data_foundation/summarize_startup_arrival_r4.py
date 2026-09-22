"""Clock-offset-independent arrival bounds and standalone HTML evidence."""
import hashlib
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/data-foundation/startup-arrival-r4'
REL=Path('results/data-foundation/2111-blocks-ui-r1/df-v001')
MIRROR=Path('/home/zjb/Desktop/worktrees/mfu-w37-v610')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    report=json.loads((BASE/'report.json').read_text());rows=report['rows'];r0=rows[0]
    qmax=max(r['barrier_end_to_ag_start_ms'] for r in rows)
    lower=max(0,qmax-r0['barrier_end_to_ag_start_ms']-r0['barrier_duration_ms'])
    upper=r0['arrival_spread_upper_bound_ms'];duration=r0['ag_duration_ms']
    residual_low=max(0,duration-upper);residual_high=max(0,duration-lower)
    latest=sorted(rows,key=lambda r:r['barrier_end_to_ag_start_ms'],reverse=True)
    assert 0<=lower<=upper and lower<=duration
    summary=dict(rank0_residence_ms=duration,arrival_wait_lower_ms=lower,arrival_wait_upper_ms=min(upper,duration),after_all_kernel_starts_lower_ms=residual_low,after_all_kernel_starts_upper_ms=residual_high,late_ranks=[r['rank'] for r in latest[:12]],assumptions=report['limits'])
    doc=f'''# 全局计时AG：是否全部在等rank到齐？r4

## 数值结论（256卡iter60）

完整提取256个rank的同一Profiler窗口内第一条全局24-FP32 AllGather，以及之前两条全局barrier。不是全量多iter扫描，没有实验、训练或预测参数变更。

- rank0 AG驻留：{duration:.3f} ms。
- 依据前一次全局barrier建立的无跨机时钟偏移假设的到达时间差界：最后一个rank的AG kernel开始，比rank0晚 **{lower:.3f}～{upper:.3f} ms**。
- 所有rank的AG kernel都开始后，rank0仍驻留 **{residual_low:.3f}～{residual_high:.3f} ms**。
- 因此，等待参与者进入AG kernel可解释rank0驻留的约 **{lower/duration*100:.2f}%～{min(upper/duration,1)*100:.2f}%**，剩余不能继续命名为“尚有rank未进入kernel”。

这不是将残余全部认定为纯传输：kernel已开始不代表内部协议/缓冲区/数据准备全部完成；残余仍可能包括协议等待、资源争用与传输。也不能把驻留100%归为到齐等待。

## 最晚的本地到达样本

表中每个时间差来自同一rank；不直接比较跨机原始ts。按barrier完成后到AG开始的本地间隔排序，不声称此排序就是严格全局到达顺序。

|rank|worker|barrier后到AG开始 ms|AG驻留 ms|CPU AG发起到GPU开始 ms|
|---:|---|---:|---:|---:|
'''
    for r in latest[:12]:doc+=f'|{r["rank"]}|{r["worker"]}|{r["barrier_end_to_ag_start_ms"]:.3f}|{r["ag_duration_ms"]:.3f}|{r["cpu_call_to_gpu_start_ms"]:.3f}|\n'
    doc+='''
## 为什么不需要把跨机绝对时间当作已校准？

对rank i，B_i/E_i为同一次前置barrier的GPU开始/结束，A_i为AG GPU开始，q_i=A_i−E_i。假设全局barrier满足“任何rank完成前，所有rank已进入”，则 B_i≤E_0 且 B_0≤E_i。

因此，最后AG开始相对rank0开始的时间差满足：

```
下界 = max(0, max_i(q_i) − q_0 − barrier_duration_0)
上界 = max(0, max_i(A_i−B_i) − q_0)
```

公式只使用各rank本地时间差，不依赖host之间的固定时钟偏移。前提：同一barrier实例、标准barrier语义、本地设备时间差可信。匹配依据为同一Profiler标签、default_pg=256、第一条24→6144全局AG及之前两次barrier；trace未提供独立后端collective序号，所以实例匹配仍有显式前提。

这给出的是“等待所有rank进入AG kernel”的时间界，不是通信算法内部每个字节何时就绪的精确分解。

## 为什么有的rank晚到：新增候选证据

实际训练日志在iteration59行后出现每rank `(after 59 iterations) memory ... max gmi memory usage`。236B参考 `training/utils.py:264` 的report_memory调用musa_patch.mem_utils.get_max_gpu0_to_7_mem_usage；补丁 `mem_utils.py:206` 同步启动外部mthreads-gmi，必要时再调用zjlab-gmi。training_log参考顺序是memory report后timers.log。

这使“显存查询/主机日志工作造成到达不齐”成为具体候选，比笼统参数量增大更贴近现有证据。没有子进程执行时间记录，且参考training.py的report_memory_flag逻辑与实际多轮日志不完全一致，不能把晚到全部归因于gmi，也不能宣称参考代码就是采集时精确版本。本任务没有执行gmi或修改日志配置。

## 建模含义与验证

保留“统计/主机准备→全局AG汇合”依赖，将rank到达差与到齐后有效成本分开。不能将rank0约1秒驻留当作96字节纯网络服务，也不能直接乘到其他通信节点。目标数据仅诊断，不更新32→256正式预测。

提取脚本audit_startup_arrival_r4.py；本文件由summarize_startup_arrival_r4.py生成。原始输入256条绝对路径、大小和SHA256见startup-arrival-r4/manifest.json；事件索引、CPU发起、GPU时间及进程组见report.json。检查256rank唯一覆盖、同一Profiler标签、消息量、两barrier和CPU/GPU关联通过。数值提取PASS；具体晚到原因及协议分离PARTIAL。
'''
    docpath=ROOT/'docs/data-foundation/STARTUP_ARRIVAL_R4.md';assert not docpath.exists();docpath.write_text(doc)
    support=[Path('/home/zjb/Desktop/fabric-data-analysis/0722/236B/Megatron-LM/megatron/training/utils.py'),Path('/home/zjb/Desktop/fabric-data-analysis/0722/236B/Megatron-LM/megatron/training/training.py'),Path('/home/zjb/Desktop/fabric-data-analysis/0722/236B/megatron-lm-musa-patch/musa_patch/mem_utils.py')]
    support+=list(Path('/home/zjb/gbs64/framework_256_gbs_64/2026-07-31-13_40').glob('worker34024/2026-07-31_1340/*.log'))
    summarypath=BASE/'summary.json';assert not summarypath.exists();summarypath.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    table=''.join(f'<tr><td>{r["rank"]}</td><td>{r["worker"]}</td><td>{r["barrier_end_to_ag_start_ms"]:.3f}</td><td>{r["ag_duration_ms"]:.3f}</td></tr>' for r in latest[:12])
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>启动AG到达等待诊断 r4</title><style>body{{font:16px/1.7 system-ui;background:#f4f6f8;color:#172638}}main{{max-width:1000px;margin:auto;padding:20px}}section{{background:white;padding:20px;margin:18px 0;border-radius:12px}}h1{{font-size:27px}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 system-ui}}.scroll{{overflow-x:auto}}a{{color:#175fa2}}.bar{{display:flex;height:35px;background:#ddd}}.wait{{background:#e4ad53;width:{min(lower/duration*100,100):.3f}%}}.rest{{background:#699bc0;flex:1}}</style><main><a href="../startup-work-r3/">← 启动工作全景</a><h1>约1秒的AG，是否全部在等rank到齐？</h1><section><h2>256个rank · iter60 · 同一次全局AG</h2><p>rank0驻留 <b>{duration:.3f} ms</b>。由前置全局barrier约束：最后参与者进入AG kernel比rank0晚 <b>{lower:.3f}～{upper:.3f} ms</b>；此后rank0仍执行 <b>{residual_low:.3f}～{residual_high:.3f} ms</b>。</p><div class="bar"><div class="wait"></div><div class="rest"></div></div><p>黄色：至少可归于尚有rank未进入AG的时间；蓝色：其余时间，含边界不确定性，不等于纯网络传输。</p><p>不能说100%是到齐等待。kernel开始也不保证内部通信协议已就绪，剩余可能含协议等待及传输。</p></section><section><h2>晚到样本（使用各rank本地时间差）</h2><div class="scroll"><table><tr><th>rank</th><th>worker</th><th>barrier后到AG ms</th><th>AG驻留 ms</th></tr>{table}</table></div><p>并未假定跨机时钟已校准。严格全局先后由文档中的barrier约束给出，不直接按原始ts排序。</p></section><section><h2>候选原因与模型处理</h2><p>实际日志包含显存报告；236B补丁会同步调用外部gmi查询。它可能造成主机侧晚到，但没有子进程耗时证据，不能确定其贡献。</p><p>建模应区分到达不齐和到齐后成本，不将整段驻留当作通信服务；正式预测未改。</p><a href="notes.md">方法与推导</a> · <a href="manifest.json">证据清单</a><details><summary>完整报告</summary><pre>{html.escape(doc)}</pre></details></section></main></html>'''
    for base in [ROOT,MIRROR]:assert not (base/REL/'startup-arrival-r4').exists()
    for base in [ROOT,MIRROR]:
        dest=base/REL/'startup-arrival-r4';dest.mkdir()
        (dest/'index.html').write_text(page);(dest/'notes.md').write_text(doc)
        for name in ['report.json','manifest.json','summary.json']:(dest/('audit-manifest.json' if name=='manifest.json' else name)).write_bytes((BASE/name).read_bytes())
        m=dict(inputs={str(p):sha(p) for p in [BASE/'report.json',BASE/'manifest.json',docpath]+support},code={str(Path(__file__).resolve()):sha(Path(__file__))},outputs={str(p):sha(p) for p in dest.iterdir()})
        (dest/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
