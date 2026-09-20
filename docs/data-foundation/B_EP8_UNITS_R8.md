# 四执行单元EP8完成结构核对 r8

2026-09-20。范围：32卡iter60 PP1 B0，rank8–15，四个checkpoint执行单元。状态PASS_STRUCTURE_AUDIT；完整B预测仍未完成。

重新核对八个原始trace哈希；根据CPU FusedCombineBackward/FusedDispatchBackward和两段_GroupedLinearBackward，使用runtime/driver correlation抽取设备事件。每单元八卡均找到两段专家分组线性反向、可见Combine后处理与本地准备设备事件。

|执行单元|最晚准备rank|有效完成残余ms|可见出口结束分散ms|
|---|---:|---:|---:|
|0|11|1.149307|0.219477|
|1|14|1.314672|0.190597|
|2|14|1.028968|0.404245|
|3|8|1.253501|0.242922|

全部满足本地准备晚于专家计算、最早可见Combine开始晚于八卡准备代理最大值，没有负残余。因此r7“本地准备→共享有效完成→各卡可见后处理”候选在四个单元均与该样本时序兼容。这不是硬barrier证明，也不表示真实网络此前未传输。

最晚准备rank发生变化，模型应由max动态计算，不能固定rank11。残余随单元变化，不应未经检验直接复制单元0成本。

仅把单元0残余代入其他单元、但使用各自实测准备时刻，边界误差分别为−0.165365、+0.120339、−0.104194ms。此为只隔离残余差异的诊断（使用了目标单元实测准备），不是source-only外推，不可用它声称预测误差小于0.2ms。

## 产物与复现

工作区`/home/zjb/Desktop/worktrees/mfu-w37-v610`。

```bash
python workflow/data_foundation/audit_b_ep8_units_r8.py --out /tmp/b-ep8-units-r8-new
```

输出必须不存在。已执行目录`results/data-foundation/b-ep8-units-r8/`包含evidence.json（32组逐rank单元证据）、summary.json、manifest.json（八个原始trace及输入/脚本/输出SHA256）。未读取iter70/256、未修改冻结预测、未提交推送。

唯一下一步：把单元0的生成器参数化到四个执行单元，保留各自trace参数并加入单元之间的同流/框架顺序，先验证四单元真正反向部分的设备边界；重计算另列，不能把它漏掉后称完整B。
