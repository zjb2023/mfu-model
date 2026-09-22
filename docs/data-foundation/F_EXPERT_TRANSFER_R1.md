# 中间F仅专家计算替换诊断

2026-09-22。固定此前B两组（完整专家计算＋已识别token/EP设备活动）替换，进一步仅替换中间F的FC1→激活/门控→FC2。其他F有效成本、Attention、CP、EP、token重排、等待残余以及外层依赖不变。

| F替换项 | 1F1B ms | 相对20396.013187ms实测 |
|---|---:|---:|
| 不替换，B两组固定 |20564.811803|+0.827606%|
| 仅FC1＋FC2 |20344.249699|−0.253792%|
| 仅激活/门控 |20512.845520|+0.572819%|
| FC1＋激活/门控＋FC2 |20292.283415|−0.508579%|

F专家子项全部替换额外减少272.528388ms，从高估168.799ms变为低估103.730ms。说明剩余存在相互抵消，不能宣称替换后误差一定单调趋零。此前中间F整块替换20.323042秒、−0.357770%仅为不同参考条件；两者差30.759ms来自在此条件下其他F成本的净作用，不是直接测得某个算子成本。

源32卡PP1 rank8 F0四层专家：FC1 13.850296ms，FC2 7.334770ms，激活/门控5.099604ms。目标256卡PP1–PP14的代表rank stage×16、每stage四个F，共56块。不是完整EP8组成本核验。

## F与B的解析区别

正常F位于checkpoint无梯度前向路径，本批trace无`_GroupedLinear`标记，因此不能照搬B重计算/真反向解析。

逐层使用明确的`_moe_permute_mask_map`结束、`aten::silu`开始、`_moe_unpermute_mask_map`开始，限定本地专家计算区间；silu前的GEMM归FC1，之后的GEMM归FC2。区间内SiLU/乘法及其CPU包含调用关联设备活动归激活/门控。只计GEMM和已识别激活，不计整个区间包络，不将重排或EP通信混入。所有关联通过同线程CPU发起和correlation，保留设备/launch事件索引；负的operator ID是推断窗口编号，不是原trace索引。该身份是结构/顺序约束推断，不是权重ID证明。

非空专家输入必须找到关联GEMM；输入为空时才允许零成本。源及每个目标F的FC1、FC2、激活并集无重复计费检查通过。源rank8沿用代表rank口径，不宣称其始终决定完整EP8关键路径。

## 复现与边界

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
python -B workflow/data_foundation/diagnose_f_expert_r1.py --out results/data-foundation/f-expert-transfer-r1-reproduction
```

输出目录需不存在。有效结果 `results/data-foundation/f-expert-transfer-r1-verified/` 含source、pp1..pp14、report、manifest；输入、代码、结果哈希及原trace路径均记录。首次`f-expert-transfer-r1/`因无GroupedLinear标记而停止，不作为有效结果。

PASS：56块、四项替换场景、B固定基线回归、关联非空检查、分项并集与哈希核验。PARTIAL：代表rank有效成本加性替换后外层DAG调度，非完整EP8内图反事实；目标iter60实测参与，−0.51%不是独立外推精度。原冻结模型和网页未修改，未提交推送。
