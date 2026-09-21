# 256卡B1：长同步窗口归属到EP Combine

## 范围与方法

iter60，256卡PP1/rank16与PP14/rank224，B1；只读两份trace并核验SHA。沿用此前B包络与归属事件，从同device所有stream事件并集排除已记录的GPU工作，再找无记录设备活动区间与runtime Synchronize调用的交集。CPU父算子按同pid/tid、完整包含调用定位，保存全部父链。

## 结果（ms）

| 无记录设备活动区间的分类 | PP1 | PP14 | PP1多出 |
|---|---:|---:|---:|
| 重计算EP Combine中的同步重叠 | 23.560 | 7.925 | 15.635 |
| 反向EP Combine中的同步重叠 | 26.676 | 12.062 | 14.614 |
| 重计算EP Dispatch中的同步重叠 | 8.469 | 8.439 | 0.030 |
| 反向EP Dispatch中的同步重叠 | 4.647 | 4.416 | 0.231 |
| 未归属选定框架算子的同步重叠 | 0.139 | 0.373 | −0.234 |
| 不与已记录Synchronize调用重叠 | 17.237 | 15.185 | 2.052 |
| 总计 | 80.727 | 48.399 | 32.328 |

两类Combine的差合计30.249568ms，约占该局部无记录活动间隔差的93.57%。已校验本样本类别交集不重复计费、分类之和与总间隔闭合。这个比例不是占整体1F1B误差的比例。

## 具体调用证据

- 重计算：父算子`FusedCombine`内的`musaDeviceSynchronize`。
- 真反向：父算子`FusedDispatchBackward`内的`musaDeviceSynchronize`；依236B语义，它调用的是Combine，而不是Dispatch。
- PP1最大的连续无设备事件窗口约14.015、11.127、8.239ms；窗口后是DeepEP `moe_unpermute_mask_kernel_tme`。前者属于真反向Combine，后两者属于重计算Combine。
- PP14对应类别也有DeviceSynchronize，最长窗口约4.740、4.649、3.061ms。之前把最大的“无本B归属事件”窗口定位到StreamSynchronize，并不意味着两stage采用不同同步机制：剔除其他stream工作后，当前分析对象已变为真正无记录设备事件的窗口。

## 如何解释

当前已从笼统的B成本差，缩小到“EP Combine上下文中的同步驻留/可见设备间隔差”。不是FlashAttention或CP计算普遍更快。同步API与间隔交叠以及后续unpermute的顺序是trace事实，但不能证明这些时间是纯CPU开销、可消除等待或纯网络传输。

这里CPU调用跨度与GPU无记录活动交集不同：之前两类Combine CPU总耗时差25.314ms，本次同步期间无设备记录交集差30.250ms，二者口径不同，并不矛盾。B整体差27.710ms还包含其他GPU活动与重叠变化。

不能将30.25ms乘PP数或microbatch数当作6.16%外推误差解释。仅两个stage、一个B；仍缺32源与目标对应成本、全组就绪证据以及DAG关键路径重调度。

## 下一步

沿236B与DeepEP实现核对Combine中DeviceSynchronize的调用位置和语义；在同一局部窗口补EP8各rank就绪/完成证据，区分跨rank不均衡、不可见通信和运行时同步开销。先验证机制，再决定是否把这一同步窗口作为独立可迁移参数。

## 复现与证据

结果：`results/data-foundation/b-sync-owners-r1/`，含逐runtime索引、CPU父链、每段前后GPU事件、时间交集和manifest。

```bash
python -B workflow/data_foundation/audit_b_sync_owners_r1.py --out /tmp/b-sync-owners-repeat
```

输出目录须不存在。未修改模型、HTML或正式预测；未提交、推送。
