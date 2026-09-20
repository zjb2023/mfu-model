# EP8反向完成耦合 r7

2026-09-20，PARTIAL_CONDITIONAL_EP8_COUPLING。32卡iter60 PP1 B0、执行单元0、rank8–15，真正反向，不包含重计算。

## 本轮结果

八卡本地Combine准备代理完成的最晚点为rank11的26.062169ms，最早可见Combine后处理开始27.211476ms。两者之间1.149307ms登记为共享有效残余，不命名为纯网络service。

```text
八路：专家梯度计算 → 本地处理/Combine准备
                       ↓ max(八卡准备代理完成)
                 共享有效残余 1.149307ms
                       ↓
             各卡入口偏移 + 可见Combine后处理
                       ↓
             梯度连接段 → Attention反向/CP
```

各卡本地准备独立推进，未加准备之前的全组屏障。共享节点是完成耦合候选，不是已证实DeepEP在这个位置执行硬barrier。准备代理终点不是已测得的payload send-ready，不能将本图解释为此前完全没有传输。

原局部Combine就绪残余被替换为共享有效残余及各卡入口偏移，不保留原残余再额外叠加。所有r6原节点结束时刻同样本回放保持一致（误差<1e-5ms）。

## 单卡成本扰动验收

对八卡各两段GroupedLinear反向分别增加1ms和10ms，共32项测试。每项均核对八卡Combine和八卡Attention结束点；另检查共享有效残余+1ms使八卡Combine均+1ms。全部PASS。

增量规律：组延迟=max(0, 单卡计算增量−该卡原本准备余量)。它是该候选在固定trace残余/固定路由条件下的结果，不是任意并行策略的定律。

- rank11专家任一段+1ms：八卡Combine和Attention均+1ms。
- rank15准备余量约7.091422ms：专家+1ms，八卡这些出口不变；+10ms，八卡均推迟约2.908578ms。

因此已可表达“一张卡专家算慢，是否及如何拖慢其他卡”，而不是无论哪卡增加成本都机械地拖慢全组。

## 范围与限制

这是明确假设下的跨卡完成模型，尚未识别pairwise payload细粒度依赖。可见permute/unpermute不等于完整通信，1.149307ms可能含同步、处理和传输，不可直接用后端纯service替换并声称物理等价。

本轮只用r6冻结中间结果，无新增原始trace扫描，无iter70/256输入。没有宣称新的跨迭代精度、完整四层B或256预测通过。旧r1–r6结果保留。

## 复现与交接

工作区`/home/zjb/Desktop/worktrees/mfu-w37-v610`。

```bash
python workflow/data_foundation/build_b_ep8_coupling_r7.py --out /tmp/b-ep8-coupling-r7-new
```

输出必须不存在。已执行目录`results/data-foundation/b-ep8-coupling-r7/`含graph、replay、pairwise-audit（实际为逐rank准备审计）、perturbation-tests、summary、manifest JSON，输入/脚本/输出哈希已登记。脚本内有基线回放、32项单卡扰动、共享残余+1验收。

唯一下一步：先在其余三个执行单元核对“本地准备→共享完成→可见Combine”的结构适用性，区分结构复用与每单元成本差异；检查是否出现负残余或先完成后就绪的矛盾，再决定四层B整合。不用目标256数据补参数。
