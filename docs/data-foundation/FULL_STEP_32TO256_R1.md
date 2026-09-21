# 源32卡首尾 → 256卡整轮及MFU r1

## 最新展示口径 · UI r3

按用户确认，现统一以ProfilerStep总时长与Profiler口径MFU为主指标。预测22.057秒、实测21.970秒、时间误差+0.40%；MFU预测4.583%、实测同口径推导4.601%。Training Step及日志残余移到默认折叠附录，不参与当前主指标拟合。以下原始r1数值记录保留供追溯，不代表现行页面主指标。

原URL `prediction-overview-r2/` 已更新。旧页面保留为 `index-ui-r2.html`；冻结r1结果不变，新UI材料在 `results/data-foundation/full-step-32to256-ui-r3/`。源代码为 `profiler_only_ui_r3.js` 和 `publish_profiler_only_ui_r3.py`；后者带版本防覆盖检查，不可直接重复发布。浏览器检查通过：主MFU正确、账本仅3项、附录可展开、手机端无整页溢出、无JS错误。

按用户要求，将启动与末B后的全部尾段改为源32卡iter60 rank0成本，倍率1。原1F1B、F/B图、PP成本不改；PP规模候选未启用。没有使用256卡首尾拟合。

## 成本与口径

| 项目 | 预测 ms | 256卡评价 ms |
|---|---:|---:|
| 启动 | 107.177209 | 1196.822985 |
| 1F1B | 21653.357840 | 20396.013187 |
| 末B后全部尾段 | 296.625584 | 376.997416 |
| ProfilerStep（以上三项） | 22057.160633 | 21969.833588 |
| 日志时钟外层残余 | 2455.488304 | 1341.966412 |
| TrainingStep | 24512.648937 | 23311.800000 |

源首尾严格使用同一rank0、同一iteration60，未混用之前多rank中位数entry/post参数。源训练日志iteration60记录14923.3ms，减源rank0 ProfilerStep12467.811696ms得到2455.488304ms残余。记录来自worker33017的RANK3日志，路径及行号3399保存在sealed-prediction.json。该残余含计时口径和可能的profiler/外层工作影响，不是已证实的纯CPU开销，也不一定是普适常数。

目标TrainingStep来自历史iteration_clocks.csv第60轮。该CSV的ProfilerStep为world包络；本评价保持rank0ProfilerStep，不混用两种边界，目标残余重新按TrainingStep减rank0ProfilerStep求得。原trace标签ProfilerStep#59与目录iteration60均保留。当前未证明整轮各rank优化器完成。

## MFU

沿用256卡历史有效FLOPs **1.293891072e17 / iter**，world_size256，峰值口径500e12 FLOP/s/GPU。

`MFU百分比 = 100 × FLOPs / (256 × 500e12 × TrainingStep秒)`

- ProfilerStep误差+0.397486%，但源启动低估约1089.646ms与1F1B高估1257.345ms抵消；不是内部成本已准确。
- TrainingStep误差+5.151249%。
- TrainingStep口径MFU预测4.123799%，实测同口径推导4.336226%，差−0.212427个百分点，相对误差−4.898895%。
- ProfilerStep口径派生MFU另列，不替代训练日志MFU。

FLOPs取自256卡历史mfu_timeline_summary.json（历史由训练日志反推的有效FLOPs），不是224卡v610常量，不是独立架构FLOPs核验；500TFLOP/s为继承口径，未新做硬件验证。数据已历史暴露，本次封存在评价前只保证不回填目标首尾，不能称盲测。状态PARTIAL：可复现经验整轮预测，不代表物理尾段模型完成。

## 资料

- 页面：http://192.168.8.16:43312/df-v001/prediction-overview-r2/
- 原r1总览保留不变。
- 结果目录：`/home/zjb/Desktop/worktrees/mfu-16to256/results/data-foundation/full-step-32to256-r1/`
- 封存：`sealed-prediction.json`、`seal.json`。
- 报告、数据检查与SHA：`report.json`、`validation.json`、`manifest.json`。
- 原数据只读，新资料未提交或推送；原模型未覆盖。

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
python -B workflow/data_foundation/predict_full_step_r1.py --out /tmp/mfu-full-step-repeat
```

输出目录须不存在。只读取源rank0一份原始trace、一个源日志、缓存和历史物理口径；目标仅既有结果评价。不重新运行大型仿真。

后续优先分清启动/日志残余的迁移误差与1F1B误差，再考虑PP1/PP2均值及少参数修正；不要以整轮0.40%误差替代分项验收。
