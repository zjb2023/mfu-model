# B侧成本块 r1

2026-09-20。32卡iter60 PP1 B0，八卡rank8–15。原始八文件哈希已核对，runtime和driver关联共同提取。每卡177个同流语义成本块，块间就绪残余另列；八卡同样本设备窗口868.833359ms。不是每kernel一个调参项。

## 重计算与真正反向必须分开

每卡4个CheckpointFunctionBackward执行单元。每单元4次重计算CP、5次反向CP，合计每卡36个CP通信设备事件；另有4个allreduce CPU通信标记，未冒充CP。COMM_wait是框架等待标记，不应当作另一项网络传输。

rank8单元0重计算消息量384、384、256、256MiB；反向256、256、384、384、256MiB（BF16输入元素数×2，组名29，均all_to_all）。这些是实际出现顺序，不预先断言对应F的CP0/CP1。用户指出CP0可能拆成两次，应进一步对照框架数学操作确认身份，不把5次硬压成4次。

框架`236B/Megatron-LM/megatron/core/transformer/moe/fused_a2a.py`明确：FusedCombineBackward调用buffer.dispatch；FusedDispatchBackward调用buffer.combine。每单元重计算中还各有一次前向Dispatch和Combine，所以四种EP语义分别登记。

## 可接成本与限制

计算包络、CP/EP有效耗时提供external_effective_replace；本地内存块及trace就绪残余保留。输入JSON使用unit=ms、basis=effective_replacement、nodes={块ID:新成本}，负数/NaN/未知或保留节点被拒绝。替换而不叠加旧B总耗时。

该版仅完成同流顺序与trace默认成本闭合；跨流事件等待及跨rank通信就绪边没有完成，所以改变成本后只是同流条件敏感性结果，**尚不能替换整段1F1B中的旧B成本作为已验证的细粒度预测**。不把同样本精确回放当独立预测。执行单元序号不是已证实的正向layer编号。

## 文件与复现

代码`workflow/data_foundation/build_b_cost_blocks_r1.py`；结果`results/data-foundation/b-cost-blocks-r1/`含graph、replay、baseline-replay、cost-bindings、evidence、summary、manifest。

`python workflow/data_foundation/build_b_cost_blocks_r1.py --out /tmp/b-cost-blocks-new`

外部输入增加`--overrides /absolute/path/overrides.json`。输出目录必须不存在。网页：http://192.168.8.16:43312/df-v001/b-cost-blocks-r1/ ，可筛选rank/执行单元/重计算或反向，编辑成本，保留参数锁定。

下一步核对两阶段CP的数学身份及跨流/跨卡依赖，之后才整合回1F1B；本轮未改变F冻结结果与旧B基线，未提交/推送。
