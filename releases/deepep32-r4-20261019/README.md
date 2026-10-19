# 新DeepEP32 → 256：同事接入入口

本目录是2026-10-19封存报告的便携发布。原始trace仍在原机器本地，**没有上传Git**。包含派生的设备活动清单、冻结DAG、评价数据和可直接运行的轻量复现器。

## 获取与查看

```bash
git clone --branch 32to256 --single-branch https://github.com/zjb2023/mfu-model.git
cd mfu-model
python3 -m http.server 8000 --bind 127.0.0.1 --directory releases/deepep32-r4-20261019
```

浏览器打开 http://127.0.0.1:8000/prediction-overview-r3/ 。本地浏览不需访问192.168.*。源设备清单、F/B图、方法比较四个页面在目录内相互链接；部分历史归档或原始trace路径为原环境引用，不属于本便携包的可用入口。

## 无原始数据的数值复现

```bash
python3 releases/deepep32-r4-20261019/reproduce.py
```

仅需Python>=3.9标准库；建议至少2GB可用内存。先校验manifest，再对371429节点的冻结DAG重新拓扑调度，使用已封存目标实测窗口重新计算20轮和末4轮误差。不会扫描trace、训练、拟合或修改文件。结果应为1F1B=22119.790913ms，主体=22422.064099ms，完整Profiler=22540.401519ms。末4轮耗时平均绝对百分比误差分别约0.707%、0.698%、5.401%。这是冻结成本的调度与评价复现，**不是从原始数据重新提取成本**。

## 模型与数据边界

- 新32卡DeepEP iter155–200成本；中间使用源PP1 F0/B0十轮对齐模板，不是PP1/PP2均值；首尾及PP通信也来自32卡。
- 目标PP16、4MB配置参与构图，256卡耗时/token不回填成本。目标历史数据已看过，不称盲测。
- 末4轮85/90/95/100是稳定尾段；全20轮主体误差约7.229%，不能省略。
- 启动统计同步计入完整Profiler结果，主体结果另列；MFU绝对数值沿用历史256卡有效FLOPs，非独立架构计算。
- 本地数据源：`/var/local/zjb-data/32-deepep-200iter-20261016/extracted/framework/2026-10-15-17:29`；目标：`/var/local/zjb-data/gbs64/framework_256_gbs_64/2026-07-31-13_40`。
- 完整源数据复现需要数据所有者提供只读访问，并适配旧worktree路径。请阅读同目录`SOURCE_AND_REPRODUCTION.md`。其中原始管线命令针对数据所有者环境，不保证在同事机器直接执行。

`manifest.json`记录便携文件哈希。原模型封存清单中的绝对路径保留用于追溯；便携校验使用新manifest。业务预测数值与原封存一致。
