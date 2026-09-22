# Profiler启动整个窗口：r2

定义：ProfilerStep开始到PP0首个F关联GPU开始；另保留首F CPU边界。不将某个AllGather等同启动阶段，不修改原预测。

| 迭代 |32卡整个窗口ms|256卡整个窗口ms|
|---|---:|---:|
|40|102.869|1260.580|
|60|107.177|1196.823|
|80|128.657|1249.176|

iter60互不重叠分段：

| 时间窗口 |32卡ms|256卡ms|
|---|---:|---:|
| Profiler起点→统计候选AG开始 |92.621|114.532|
| 统计候选AG设备驻留 |2.210|1068.895|
| AG完成→首F CPU |7.738|9.069|
| 首F CPU→首F GPU |4.609|4.328|

每行仅为时间边界分段，不宣称单一物理成本。总启动差1089.646ms，其中候选统计AG驻留差1066.685ms，占约97.9%；其余时间合计源104.968ms、目标127.928ms。意味着差异主要位于首F之前的全局活动，不是首F设备启动突然变慢；不证明其全部为到齐等待。

源/目标每rank此AG输入均96字节；60%带宽无法解释秒级驻留。256卡跨三轮窗口稳定在约1.20–1.26秒，而32卡约0.103–0.129秒；规模关联明显，但采集日期、层数、软件/运行条件不同，不能拟合独立rank数量因果公式。

236B training.py约2066行在训练循环开头调用prof.step，后续有异步保存清理、microbatch检查、train_step等；约2198行training_log和timers.log位于训练后。实际trace却在首F之前已有统计形状AG，说明不能仅凭参考代码把它命名为“上一轮统计”。需要实际采集代码/调用栈或日志边界确认。ProfilerStep#59对应数据目录iteration_60，原编号均保存。

后续模型接口建议：startup = 本地准备有效项 + 全局统计/同步有效项 + 首F发起项。通信带宽只描述明确传输，不把整个启动窗口按rank倍数或链路速率缩放。目标数据先只用于诊断，不自动回填约1.1秒。

## 尾段简化候选（与启动分开）

源296.625584ms＋DP RS延长9.932459ms＋EDP RS31.457280ms＋两轮AG暴露增量10.115482ms=348.130805ms，再加独立同步开销。相对目标376.997416ms仍差28.866611ms，不直接命名为同步。

基于400Gb/s×60%=30GB/s、目标同规格/独享lane/DP分层、源机内与普通AG成本固定、PP0 EDP RS全暴露等假设。AG两轮分别max(DP,EDP)，不累加各PP stage，源有效AG含等待，故仍是简化候选。

## 复现与验证

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
python -B workflow/data_foundation/audit_profiler_startup_r2.py --out results/data-foundation/profiler-startup-r2-reproduction
```

只读6份rank0 trace，结果`results/data-foundation/profiler-startup-r2/`含路径、SHA256、Profiler标签、CPU/GPU边界、四段守恒及设备/同步并集。运行时同步与GPU活动重叠，禁止相加。未记录设备时间不等于GPU物理空闲。

PASS：六样本边界顺序、分段总和和既有iter60启动值一致。PARTIAL：全rank最后就绪、统计调用身份、rank规模独立因果关系未完成。未训练、未拟合、未提交推送。
