# 单层 CP/EP 主干冻结 r3

2026-09-20。冻结范围是32卡 iter60 / PP1 / F0 / layer0 / rank8–15的同样本条件主干；不是独立预测精度验收，不改变v610与224卡版本。

产物：`/home/zjb/Desktop/worktrees/mfu-w37-v610/results/data-foundation/ep8-backbone-r3-frozen/`。manifest登记代码、父版本、原始八文件哈希及输出哈希。本次为内容哈希冻结，不作Git提交/推送。r1、r2产物保留。

## 层尾

原始trace八卡各两个AddOp设备kernel，位于Combine可见unpermute结束后、下一次LayerNorm之前。新增本地交接、两个Add和下一层入口交接节点，各rank独立推进。终点取八卡下一层LayerNorm开始的最大值，不执行下一层计算；最后一个Add结束也单独登记，以区分本层设备工作与层间交接。框架transformer_layer.py的MLP后残差相加支持该局部归属，但不逐个宣称两个Add的具体张量语义。

## 成本合同

`cost-bindings.json`逐节点登记默认trace值和权限：

- `external_effective_replace`：Attention本地块、专家相关块、层尾Add、CP有效成本、各卡Dispatch有效残余、公共Combine有效残余。外部值替换原值，不相加。计算包络不等于纯算术kernel成本；纯计算输入也需要先拆除内部间隙。
- `trace_reserved`：输入重排暴露尾段、本地准备、就绪同步、交接残余、Combine可见unpermute及各卡完成偏移等。当前配置默认沿用trace。

注入文件示例由`override-example.json`提供：`unit: ms`、`basis: effective_replacement`、`nodes: {combine_residual: 新值}`。拒绝负数、NaN、未知节点及保留节点覆盖；纯网络service口径不接受直接注入。

## Combine 如何接入

先取八卡专家相关块完成的最大值。公共`combine_residual`默认 **0.9696331024 ms**，即“最晚专家块结束→最早可见unpermute开始”；之后各卡保留自己的完成偏移与unpermute时长，再接本卡层尾。

公式：`ready=max(expert_end[r])`；`common_end=ready+C_combine_effective`；`local_unpermute_end[r]=common_end+trace_offset[r]+trace_unpermute[r]`。不得再把整个FusedCombine CPU包络相加。各rank的等待由max产生，不再累计收费。

该接口是有效成本，不是已识别的传输service。若外部只提供纯传输时长，目前缺少把旧0.969633ms拆成网络、软件与残余的依据，不能声称直接替换就有物理精度。CP有效成本和重排暴露尾段同样有重叠依赖，拓扑/通信成本大幅改变时需重新拆分。

## 验证与复现

生成脚本`workflow/data_foundation/freeze_ep8_backbone_r3.py`校验八原始文件哈希、每卡两个Add、原56边界不变、八卡下一层入口吻合，并测试Combine+1ms导致总结束只+1ms。

```bash
python workflow/data_foundation/freeze_ep8_backbone_r3.py --out /tmp/ep8-r3-new
python workflow/data_foundation/freeze_ep8_backbone_r3.py --overrides results/data-foundation/ep8-backbone-r3-frozen/override-example.json --out /tmp/ep8-r3-override-new
```

输出必须为新目录。网页读取冻结graph/replay，保持旧逻辑图及演示成本模型分区，不将二者假装同一套校准参数。
