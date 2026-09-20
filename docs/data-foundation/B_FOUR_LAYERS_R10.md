# 四层B混合DAG r10

2026-09-20。状态PASS_HYBRID_REPLAY_PARTIAL_PREDICTOR。

范围为32卡iter60 PP1 B0、rank8–15，四层按stage内第4→1层执行；每层先重计算、再真正反向。前向/反向checkpoint Sequence number在rank8已核对为24281/24293/24305/24317的正序与逆序，层名称为本stage内顺序而非全模型编号。

## 已完成

- r5/r6生成器增加unit参数，r7支持指定源目录。单元1–3使用各自trace证据生成，不复制单元0的成本。
- 四层MoE反向、EP8条件完成耦合、梯度连接段、Attention反向及CP2条件耦合接入r9装配骨架。
- 原真正反向包络被细粒度子图替换；未覆盖头尾单独保留，绝不同时叠加旧包络总耗时。不同rank各自沿层序推进，不添加额外层间全rank barrier。
- 64个阶段边界、4238个细粒度子图节点结束点检查通过；完整所选B设备窗口868.833359ms与原基线一致。4238包含就绪/连接节点，不是4238个独立拟合参数。
- 64组专家成本扰动（每层每卡两段GroupedLinear），分别检查+10ms及+50ms。+10有2组被余量完全吸收；+50全部传播到该层八卡及后续各层、最终B出口，B增加38.464822～50ms。

## 保留范围

重计算仍按各层trace包络保留，反向头尾及大量本地就绪/处理残余也保留，未完全继承F内部成本化。CP/EP共享有效残余并非纯网络service；跨卡完成节点与部分本地跨流边仍是条件建模假设。这里完成的是全窗口混合装配与回归，不能称完整物理参数模型或独立预测通过。

未读取iter70或256做本轮拟合，原v610/224模型及历史冻结结果保留。无Git提交、推送。

## 复现

工作区`/home/zjb/Desktop/worktrees/mfu-w37-v610`。已有单元0封存输入保留；其他单元输入如下生成（输出均须不存在）：

```bash
for u in 1 2 3; do
  python workflow/data_foundation/build_b_gradient_bridge_r5.py --unit "$u" --out "results/data-foundation/b-four-layer-r10-u${u}-bridge" &&
  python workflow/data_foundation/build_b_moe_backward_r6.py --unit "$u" --bridge-dir "results/data-foundation/b-four-layer-r10-u${u}-bridge" --out "results/data-foundation/b-four-layer-r10-u${u}-moe" &&
  python workflow/data_foundation/build_b_ep8_coupling_r7.py --source "results/data-foundation/b-four-layer-r10-u${u}-moe" --out "results/data-foundation/b-four-layer-r10-u${u}-ep" || break
done
python workflow/data_foundation/build_b_four_layers_r10.py --out /tmp/b-four-layers-r10-new
```

已经存在的上述单元目录无需重建；直接最后一条可重验装配。实际结果目录`results/data-foundation/b-four-layers-r10/`：graph/replay/coverage/tests/summary/manifest JSON。各单元另存各自输入/输出哈希。

## HTML

http://192.168.8.16:43312/df-v001/b-four-layers-r10/

提供四层总览、逐rank边界、单层功能流程、真实成本节点与前驱、有效成本编辑和完整重置。原F页面保留并增加新入口。图中的流程框解释模块，真实计算来自发布的完整DAG，非独立手写时间线。

发布脚本`publish_b_four_layers_r10.py`；模板`ui_b_four_layers_r10.html`；浏览器测试`test_b_four_layers_r10.cjs`。已通过层/rank切换、专家+50传播、负值拒绝、重置、390px无整页横向溢出及无JS错误；证据`results/data-foundation/b-four-layers-r10-ui-check/`。沿用lark-apps技能的可读性与本地页面检查，按用户范围仅在43312发布，无云端资产创建。

唯一下一步：封存本版iter60输入/成本/图，明确保留项后提取同口径iter70 B0各层及整段边界进行评价；目标数据只进评价器，不回填参数。iter70已有历史曝光，不能称盲测。
