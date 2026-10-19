# 新DeepEP32 → 256 DAG：2026-10-19结论封存

## 状态与范围

本次封存已有模型、报告及展示，不是新实验。工作区 `/home/zjb/Desktop/worktrees/mfu-16to256`，分支 `32to256`，基底提交 `7f95417996d65846bf94fba2ad4ab3e0dab231c1`。工作区有未提交改动，不能只用该SHA代表此版；以归档manifest逐文件SHA256为准。本次不提交、不推送。

报告入口：http://192.168.8.16:43312/df-v001/prediction-overview-r3/

源成本固定，所有目标轮次使用同一预测。目标末4轮为85/90/95/100：

|范围|预测秒|实测均值秒|耗时平均绝对百分比误差|预测MFU|实测推导MFU均值|MFU平均绝对百分比误差|
|---|---:|---:|---:|---:|---:|---:|
|完整Profiler窗口|22.540|23.828|5.401%|4.485%|4.242%|5.710%|
|训练主体：1F1B＋尾部|22.422|22.580|0.698%|4.508%|4.477%|0.705%|
|1F1B|22.120|22.090|0.707%|—|—|—|

主体平均少估约158ms：1F1B高估29.6ms、尾部低估187.5ms，存在抵消。全部20轮主体误差7.229%、1F1B误差8.044%，不能将末4轮精度推广到全程。MFU按每轮耗时分别换算后求均值。

Profiler前缀包含步间统计与同步。236B参考代码中日志计时器barrier和计时汇总小AG由日志配置触发，不依赖profiler开关；不能将全部前缀视为采集额外开销。Profiler与训练日志边界不同。

## 数据来源与用途

- 新源32卡：`/var/local/zjb-data/32-deepep-200iter-20261016/extracted/framework/2026-10-15-17:29`。DeepEP、PP4、CP2、EP8、EDP1。稳定性观察iter5–200共40轮；成本使用155/160/165/170/175/180/185/190/195/200。
- 中间F/B：源PP1、rank8–15、F0/B0四层子图，十轮语义对齐后平均成本再调度；不是PP1/PP2均值。PP2只参与稳定性观察。
- 首尾成本和PP通信：同源rank0/8/16/24缓存；PP0、PP3映射目标PP0、PP15。前缀118.337ms和尾部302.273ms来自源rank0十轮均值。
- 目标256卡：`/var/local/zjb-data/gbs64/framework_256_gbs_64/2026-07-31-13_40`；配置PP16、4MB、EDP2用于构图。iter5–100每5轮一次共20轮仅用于事后评价/误差诊断。当前成本没有回填目标token或耗时；历史目标已见过，不是盲测。
- MFU绝对数值例外：有效FLOPs=1.293891072e17，峰值500TFLOP/s/卡，继承 `/home/zjb/Desktop/mfu-model/case_256gpu_pp16_cp2_a2a/results/mfu_timeline/mfu_timeline_summary.json` 的历史256卡日志推导口径，非本次独立架构推导。FLOPs不参与耗时调度。
- 历史尾部60%带宽方案使用目标消息规格，但当前主报告没有加入该增量；不要将profiler-completion-r1/sealed-prediction.json的历史条件候选误认为当前展示预测。当前值以prediction-overview-r3/scope-comparison.json为准。

所有原始trace不入归档。精确逐文件路径和原始哈希见 `results/data-foundation/deepep-stable-20261016-r1/inventory.json`、各costs-r2完成清单、outer.json、profiler-completion-r1/manifest.json；目标输入见evaluation.json。旧32卡组元数据和外部GPU边界提取器仍是依赖，不是独立可搬迁执行包。

## 复现方法

先校验封存包（manifest在包内），不要在冻结目录中覆盖结果。以下命令在上述工作区或其独立副本根目录执行。

### 只重建当前网页（不重跑模型）

```bash
python3 workflow/data_foundation/publish_new_deepep_suite_r1.py
```

读取既有封存、评价和诊断缓存，生成总报告、清单、F/B图；也会发布到现有43312镜像目录。隔离复现时须先把脚本WEB改到新的测试发布目录，不覆盖线上。本次已执行并浏览器检查无JS错误。

### 源数据至DAG的管线

```bash
/var/local/zjb-data/mfu-tools/snakemake-venv/bin/snakemake -s workflow/deepep_stable/Snakefile-r4 --cores 2 --resources mem_mb=12000 --dry-run
# 检查输入和输出路径后，在独立准备的工作副本运行，去掉 --dry-run。
```

必须使用Snakefile-r4，不是默认旧Snakefile。读取新32卡数据→token稳定性→逐轮F/B提取→r4语义对齐→首尾/PP成本→模型封存→256评价。该管线结束后还需运行发布器生成本次总报告。缓存带哈希，不能盲目强制覆盖；原路径依赖及环境见environment.txt、config.json、manifest。本次未重跑此管线。

### 利用已冻结源缓存重新构图、评价

```bash
python3 workflow/data_foundation/align_deepep_stable_r4.py
python3 workflow/data_foundation/model_deepep_stable_r4.py seal
python3 workflow/data_foundation/model_deepep_stable_r4.py evaluate
```

仅在独立副本使用；会重写aligned-r4/model/evaluation输出。期待F=350.893123ms、B=886.621257ms、1F1B=22119.790913ms，112个中间子图实例。模型先封存，评价后核验封存不变。主体/完整窗口展示再加源端尾部/前缀，未经目标拟合。

## 限制

当前是代表EP8组及rank0窗口评价，不是256rank完整makespan证明；目标两组EP8假设等价。有效通信成本含等待，条件依赖部分用源就绪残差保留，不能视为所有因果边已证实。源成本迁移不能预知目标路由工作量。绝对MFU与耗时预测的数据依赖不同。

归档脚本 `workflow/data_foundation/freeze_deepep_report_20261019.py` 生成唯一时间戳目录、manifest与tar.gz；保存当前代码、配置、模型、源缓存、四个页面和展示输入依赖，原始trace仅引用。它不会重跑模型或修改Git。
