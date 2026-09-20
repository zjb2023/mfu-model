# 32卡细化F/B装配 r1

2026-09-20。当前工作区 `/home/zjb/Desktop/worktrees/mfu-w37-v610`。
状态：装配/封存回归PASS；全世界逐rank物理模型仍PARTIAL，未设精度通过阈值。

## 完成

1. 从32卡iter60九份已有trace核对PP1参考rank8和PP2全部rank16–23：四层EP/checkpoint/GroupedLinear计数、CP2/EP8组规模、F16次/B36次CP有序消息签名、专家矩阵尺寸一致。只证明观察签名兼容，不证明全部跨流依赖或所有microbatch等价。PP2没有独立细化成本校准。
2. 以PP1的四层F图和B r10图为模板，在PP1/PP2各8microbatch装配32个子图实例，实际展开83617节点。每个旧中间F/B包络由零成本出口替换；入口继承外层调度依赖，出口依赖子图END。没有旧包络与子图重复计费，也没有额外叠加CP/EP通信总耗时。
3. 首尾stage保留原source60角色标量；独立PP有效成本沿用source60；最后stage每个F末尾Loss预留值0。更新尾段RS/OPT/AG不进入本轮范围。
4. 全部外层节点结束点与等价收缩图一致；32个模块跨度匹配独立模板；专家成本+1000ms压力测试传到整段窗口（+997.806229ms）。这验证接线，不是实际性能预测。
5. 读取目标缓存评价之前封存图、参数与预测，评价后哈希不变；没有70成本拟合，也没有256新预测。

## 结果

| 口径 | 预测ms | 实测ms | 有符号误差ms | ARE |
| --- | ---: | ---: | ---: | ---: |
| 32卡iter60，PP0首F→末B | 11917.227759 | 12064.008904 | −146.781144 | 1.216686% |
| 32卡iter70，同口径 | 11917.227759 | 12260.027324 | −342.799564 | 2.796075% |

中间模板固定F342.713883ms、B868.833359ms。60→70实测同口径增加196.018420ms（约1.624820%）。这里是包含流水线填充/排空的FB+PP区间，不是ProfilerStep，也不是之前B0的2.79%结果。

## 边界与保留假设

- 外层以stage组级完成衔接PP，继续使用已有简化rendezvous，并非全32rank逐一PP完成/重叠的精确复现。子图内部rank归属映射PP1→8–15、PP2→16–23。
- 同一份F0/B0参数用于所有microbatch和两个中间stage，是此次显式迁移假设；source trace就绪残余及重计算包络保留。
- PP2源F0约345.641～346.650ms，B0约878.915～881.749ms（各rank本地窗口），不完全等于PP1成本；仅诊断，未用这些时间调节模板。
- 目标评价沿用既有runtime-only设备归属缓存；细化模型使用runtime+driver归属。这个口径差异已保留，不能声称已做完整逐事件对齐。
- 主指标以rank0设备区间为准，其他代表rank0/8/16/24时间线是诊断；跨机时间可比性有假设，非全世界makespan。
- iter70已经历史曝光，非盲测；上述误差不是证明细化模型优于旧模型。

## 路径和复现

`results/data-foundation/pp2-module-compatibility-r1/`：evidence/summary/manifest。
`results/data-foundation/pp32-detailed-fb-assembly-r1/`：展开graph/replay、32项bindings、parameters、module-windows、sealed-prediction/seal、report/timeline/checks/manifest。

```bash
python -B workflow/data_foundation/audit_pp2_module_compatibility_r1.py --out /tmp/pp2-module-check-new
python -B workflow/data_foundation/assemble_pp32_detailed_fb_r1.py --out /tmp/pp32-detailed-fb-new
python -B workflow/data_foundation/publish_pp32_detailed_fb_r1.py
node workflow/data_foundation/test_pp32_detailed_fb_r1.cjs results/data-foundation/pp32-detailed-fb-ui-check
```

输出必须不存在；装配器使用已保存的PP2核验目录，发布器使用已保存装配目录；已有发布不覆盖。
网页：http://192.168.8.16:43312/df-v001/pp32-detailed-fb-r1/

下一步：在新的版本中保留首尾角色并扩展中间2→14stage，按256实际microbatch4重新调度；先封存预测再评价256同口径区间，不做token拟合，不加入更新尾段。沿用的stage聚合近似及PP跨机成本迁移假设必须继续列明，不能默认为已验证。

未改旧图/旧预测/原始数据，没有Git提交或推送。

浏览器检查PASS：HTTP200、预测值、60/70切换、128个正宽度F/B条块、390px无整页溢出、无JS错误。截图已查看，证据 `results/data-foundation/pp32-detailed-fb-ui-check/`。所有登记的输入、代码、输出SHA256复核通过。
