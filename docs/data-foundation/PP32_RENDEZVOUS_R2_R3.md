# 最简32卡PP链：通信会合与GPU边界修正

2026-09-18。原v610、224卡、2111及pp32-minimal-r1保持不变。未提交、推送或运行训练。
代码及结果工作区：`/home/zjb/Desktop/worktrees/mfu-w37-v610`。

## 结论

保留每stage每微批一个F/B大块，不引入kernel级预测参数。
补充PP收发会合规则，并把F/B成本边界改为关联GPU事件的首开始至末结束。
源60校准、70评价，四rank PP链的预测完成时间对CPU ProfilerStep窗口均值误差：

| 模型 | 60误差 | 70误差 | 含义 |
|---|---:|---:|---|
| r2 CPU块＋有效通信会合 | 约4.53% | 约2.78% | 反向PP成本混入B GPU尾部 |
| r3b GPU块＋零通信耗时诊断 | 1.229% | 2.878% | 有会合等待，但传输耗时设零，仅作消融诊断 |
| r3b GPU块＋有效通信会合 | 0.787% | 2.444% | 当前条件性试点结果，非全32卡世界时长验证 |

r3b预测12369.693ms，60实测四rank ProfilerStep均值12467.867ms，70均值12679.532ms。
预测值只来自60；两个目标使用同一参数图。70已在旧版本公开评价，本轮不称盲测。
不能把该误差称为MFU误差；未算Training Step或256卡MFU。

## 为什么改图而不是加大量参数

只用F→下游F边，不足以描述阻塞发送方何时继续工作。每个有效PP调用加入就绪与完成节点；一条消息在发送和接收双方就绪后执行，调用完成才释放该stage的后续任务。
双向收发调用必须等待其中两条消息全部完成。等待由max依赖计算，不另拟合等待成本。
每轮这条链64个F/B节点、48条消息，辅助节点无耗时。

发现CPU B返回后，stage1/2/3仍有约93—97ms关联GPU活动；这些活动与PP等待窗口重叠。
按CPU调用窗口直接计算通信会合交集时，反向中位数95.697ms；先扣除端点此前GPU尚未完成的部分后，反向有效交集为2.725ms。
前向有效交集中位数2.762ms。只采用收发两端均为单向调用的6个前向、6个反向消息样本；双向组合调用不用于拟合，避免把另一条消息等待混进成本。
这些数值仍包含软件同步，且依赖跨主机时间戳可比的假设，不称独立测得的纯网络service。

## r3b参数表

| 参数 | ms |
|---|---:|
| 首stage F / B | 163.476 / 430.892 |
| 中间stage共享 F / B | 344.115 / 873.501 |
| 尾stage F / B | 205.522 / 501.034 |
| 前向 / 反向有效PP | 2.762 / 2.725 |
| 共同启动包络 | 101.277 |
| 全链完成后的包络 | 297.967 |

总计10个经验耗时参数，其中仍只有6个F/B参数。额外4项中启动及末尾包络是已知不足：不是可直接外推的物理常数，尤其不能把末尾包络叫成完整OPT模型。
当前尾段采用全链结束后一个共同包络，未还原DP bucket重叠或各stage独立更新。不会据此自动外推256。

## GPU关联与失败版本

每个GPU事件通过唯一runtime correlation找到CPU发射事件，再按同进程CPU F/B时间包含关系归属，允许不同线程。
同一GPU事件只归属一个F/B；验证128个块的GPU包络互不重叠。这比r1只允许同线程的覆盖更完整，但跨线程因果身份尚未由框架sequence/flow逐项证明。
**不要求识别每个kernel用途；这些关联仅用于测量大块边界。**

初次r3尝试混用“CPU进入至GPU完成”作为一个块，源60 rank0即发现相邻块重叠：CPU已提前发射下一F。程序在校准前停止，未读取70。失败协议保留在 `pp32-gpu-boundary-r3/`。
修正为GPU首尾边界后另起r3b目录，没有覆盖失败结果。

## 工程验收与证据

- r2：`results/data-foundation/pp32-rendezvous-r2/`。
- r3b：`results/data-foundation/pp32-gpu-boundary-r3b/`。
- `source-pp-pairs.json`：48个消息配对与会合窗口证据。
- `boundary-audit.json`：逐rank、逐相位GPU完成与CPU结束的偏移。
- `predictions.json`：两种模型的完整图、成本和时刻。
- `prediction-seal.json`：本轮读取70数据前已写入预测及参数哈希；这是执行顺序约束，非操作系统级数据访问隔离。
- `independent-check.json`：21个文件哈希，476个节点递推、128个F/B归属和48消息检查PASS；不代表时钟或物理假设验证通过。
- 独立新目录 `results/data-foundation/pp32-gpu-boundary-r3b-reproduce/` 重跑及独立核验PASS；`reproduction-check.json` 确认参数、预测、评价、消息对、边界及两轮观察共7个结果文件逐字节一致。

```bash
cd /home/zjb/Desktop/worktrees/mfu-w37-v610
.venv/bin/python -B workflow/data_foundation/pp32_gpu_boundary_r3.py --out /tmp/pp32-r3b-NEW
.venv/bin/python -B workflow/data_foundation/check_pp32_rendezvous.py /tmp/pp32-r3b-NEW
```

输出目录必须不存在。只读既有r1记录和同样8份原始trace；不读256配置或时序，不调用v610成本。

## 状态与下一步

整体仍为 **PARTIAL**，不是256外推已完成。
下一步先覆盖其余28rank，验证三类F/B模板和48消息的会合规律能否代表全部8条PP链；同一GPU不可重复计费，尾段等待与更新须分开登记。
再将图生成从源调用模板转成目标配置驱动的标准1F1B，并核对DP4→8、微批8→4和网络布局；入口及尾段包络不得原样冒充可迁移常量。
