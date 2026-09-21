# 224/256中间B：位置规律与跨迭代稳定性

2026-09-21，只读缓存和9份原始trace，不修改模型。用户提醒224数据位置有软链接；按历史source_path读取，登记resolved_path，不搬迁数据。此轮使用的fabric-data-analysis原始路径可读，不依赖重新复制。

## 结论

不能把“256 iter60中间B都低于源32卡868.833ms”推广为稳定跨迭代规律。

相同GPU归属口径抽查：

| world | stage | iter60 B均值 ms | iter100 B均值 ms |
|---|---|---:|---:|
| 256 | PP1 | 818.044 | 870.146 |
| 256 | PP7 | 798.502 | 895.091 |
| 256 | PP14 | 809.272 | 870.025 |
| 224 | PP1 | 1028.193 | 1110.388 |
| 224 | PP6 | 859.037 | 928.573 |
| 224 | PP12 | 980.472 | 1138.158 |

256 iter100三个样本都不再低于源B模板，其中PP1/PP14仅略高，PP7明显更高；224抽查显示前段和后段较长，中段较短，不能套用“所有中间stage近似且普遍更短”。224每stage3个microbatch、256每stage4个，此表是块时长均值，不是一个stage全轮B求和。

跨iter CPU旁证（必须与GPU包络区分）：对224/256分别检查lane0、全部中间stage，iter60,65,70,75,80,85,90,95,100九轮。224每轮中间B CPU平均值由828.921ms升至924.330ms；256由709.597ms升至782.114ms。CPU缓存标注forward_step/backward_step，不是GPU完成时间。224每轮PP1–5和PP12均值均高于PP6–11中的最大stage均值（基于本次缓存表）；此现象尚不能归因为层身份、设备或负载。

因此当前应同时考虑“规模”和“迭代/运行状态”，不能仅用PP规模系数解释误差。尚未定位漂移来自重计算、专家token、通信、频率或profiler开销，不作因果断言。此轮GPU只抽查9份trace，不能声称所有迭代、所有stage或所有rank都已证明。

## 边界与复现

GPU抽查：224 iter60/100各PP1/6/12；256 iter100 PP1/7/14。采用与现模型相同的唯一runtime correlation、CPU包络内关联GPU首尾方法。256 iter60取既有诊断stage缓存。原始路径、软链接解析后路径、SHA、CPU和GPU事件索引均在结果中。

CPU数据来源fab工程现有stage/lane事件缓存；未采用跨rank stage总包络，也未混用CPU与GPU数值拟合。

结果：`/home/zjb/Desktop/worktrees/mfu-16to256/results/data-foundation/b-scale-stability-r1/`，含report、9份GPU证据及manifest。

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
python -B workflow/data_foundation/check_b_scale_stability_r1.py --out /tmp/mfu-b-stability-repeat
```

输出目录须不存在。本次未改HTML、未提交或推送，未校准新系数。

下一步应固定同口径迭代集合，再检查源32卡60/70与目标224/256多轮B分项；不能只针对目标iter60消除高估。
