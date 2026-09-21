# 基于236B语义的B框架算子位置对照

## 已执行范围

沿用上一轮4份trace：224卡iter60 PP5/6、256卡iter60 PP1/14，各stage lane0代表rank，全部14个B。未读取32卡，未修改预测参数。每B识别40个选定框架算子调用，共560个，均找到至少一个唯一runtime correlation关联的设备事件。

依据236B参考源码：

- `transformer_block.py:456`附近block重计算对layer分别checkpoint，B包含重计算和真反向。
- `moe/fused_a2a.py`：FusedDispatch前向调用buffer.dispatch，并明确注释CPU等待GPU信号；FusedCombine前向调用buffer.combine。FusedCombineBackward实际调用dispatch，FusedDispatchBackward实际调用combine；async_finish路径有current_stream_wait，是否走该分支需采集配置确认。
- `moe/experts.py:858`及921–927：FC1→激活→FC2；两次GroupedLinearBackward按逆序暂标FC2/FC1，未做权重ID核验。
- trace出现AttnFuncWithCPAndQKVOA2A及其Backward，用作Attention+CP的整体算子标记，不能拆称纯Attention计算。

参考代码树：`/home/zjb/Desktop/fabric-data-analysis/0722/236B/Megatron-LM/megatron/core/transformer/`。manifest记录代码哈希，不宣称该参考树就是224/256采集时的精确安装版本。

## 统计口径

按CPU框架标记统计调用次数、CPU累计包络耗时，以及同线程runtime关联设备事件的并集和逐调用GPU包络。CPU耗时包含发射、框架处理和可能的等待，不等于GPU算力成本；设备可见事件也未必覆盖全部DeepEP传输。各类别可以重叠，不能求和当作B壁钟归因。重计算、真反向分别标记，但当前不是全算子覆盖，仅选定40个上层标记。

## 示例差异：后stage减前stage，单位ms

| 对照 | B | 框架功能 | CPU累计变化 | 关联GPU活动并集变化 |
|---|---:|---|---:|---:|
| 224 PP5→6 | 0 | Attention+CP反向 | −26.62 | −7.92 |
| 224 PP5→6 | 0 | 专家FC1反向（顺序推定） | −16.88 | −1.62 |
| 224 PP5→6 | 1 | 重计算EP Dispatch | +44.20 | −53.44 |
| 256 PP1→14 | 1 | 重计算EP Combine | −13.09 | −0.14 |
| 256 PP1→14 | 1 | 反向EP Combine | −12.23 | −0.14 |
| 256 PP1→14 | 2 | 反向EP Combine | −13.06 | −0.16 |
| 256 PP1→14 | 3 | 反向EP Combine | −11.93 | −0.09 |

具体解释：256 B1的两个Combine框架调用显著缩短，但关联可见GPU活动变化很小。与上一轮观察到的B事件间隔减少相容，提示框架调用驻留/同步关系值得调查，不支持直接归为专家矩阵计算普遍加速。不能将两个CPU差值相加，断言它们解释了B壁钟下降。224 B1 Dispatch的CPU和GPU变化甚至方向相反，再次说明两种时钟口径不能混用。

## 当前结论与下一步

已做到框架算子层的首轮对照，而不是仅按kernel名字比较。主要线索是EP相关框架调用驻留时间和算子间隔变化；尚未证明是哪个跨rank依赖、主机处理、stream同步或硬件执行因素造成。

下一步优先拆256 B1的两个Combine调用：runtime同步API、关联设备完成和其他stream覆盖；必要时再补同EP组rank，核对到齐等待。不把CPU调用耗时直接放进GPU DAG，也不据此宣称6.16%误差已经物理归因。

## 结果与复现

- 结果：`results/data-foundation/b-framework-position-r1/`，每算子保留CPU事件索引、设备事件索引、时长、原trace与参考代码SHA。
- 代码：`workflow/data_foundation/audit_b_framework_position_r1.py`。
- 状态：PARTIAL_FRAMEWORK_MARKER_ATTRIBUTION；4份输入SHA核验、每B 40调用检查通过，无调用完全缺少设备关联。
- 未修改HTML、冻结模型，未提交或推送。

```bash
python -B workflow/data_foundation/audit_b_framework_position_r1.py --out /tmp/b-framework-position-repeat
```

输出目录须不存在。
