# 优化器尾段 r1：首轮事实拆分

F/B误差分析已冻结于本地提交 `8e55d28`，分支32to256，未推送。该冻结版本解释了iter60大部分1F1B高估，但不代表B物理误差来源已闭合。

本轮后续任务：只读源32/目标256各一份iter60 rank0原始trace，按同一GPU归属和末B边界核对优化器尾部。

| 观测窗口 | 32卡 ms | 256卡 ms | 目标−源 ms |
|---|---:|---:|---:|
| 末B→最后RS完成 | 33.242346 | 95.359826 | +62.117480 |
| 最后RS→首个优化器关联GPU | 28.472837 | 26.290431 | −2.182407 |
| 首个优化器关联GPU→最后AG完成 | 80.266528 | 84.688796 | +4.422267 |
| 最后AG→ProfilerStep结束 | 154.643872 | 170.658364 | +16.014492 |
| 总尾段 | 296.625584 | 376.997416 | +80.371832 |

四个窗口严格首尾相接，总和通过验证，但只是实测端点分段，不意味着窗口内操作串行或某段只包含一种物理成本。CPU FusedAdam包络内runtime关联的设备事件用于定位更新开始，跨线程因果仍为条件归属。

另按RS、AG、AllReduce、优化器关联GPU、梯度处理、拷贝与其他设备活动统计并集，保存逐事件证据。不同类别可能重叠，不能将各类并集求和代替壁钟时间。AllReduce按名字分类，不把所有小allreduce认定为纯同步。此版未区分每个RS/AG的DP或EDP身份。

## 验证与边界

原始trace SHA与此前来源匹配；四段总和与末B→ProfilerStep跨度一致。无全量扫描、无训练/完整仿真、无目标成本回填；只有rank0、一个迭代，不称全世界优化器尾段已建模。

网页：http://192.168.8.16:43312/df-v001/optimizer-tail-audit-r1/

结果 `results/data-foundation/optimizer-tail-audit-r1/`，网页 `results/data-foundation/2111-blocks-ui-r1/df-v001/optimizer-tail-audit-r1/`。

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
python -B workflow/data_foundation/audit_optimizer_tail_r1.py --out /tmp/optimizer-tail-audit-repeat
```

输出目录必须不存在。发布脚本为一次性创建，不重复覆盖。

## 唯一下一步

优先核对源/目标RS的DP/EDP组别、消息量、两端就绪与GPU完成，解释末B→最后RS窗口增加的62.117ms。先不改AG/更新参数，不拿目标窗口差直接拟合带宽。源尾段可先以四个可加预留窗口表达，源值之和与原预测保持一致。
