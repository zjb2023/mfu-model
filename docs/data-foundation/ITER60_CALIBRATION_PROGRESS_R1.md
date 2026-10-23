# 256卡iter60详细DAG标定：进展与复现

日期：2026-10-23。目标尚未完成；本页的局部PASS不是完整MFU验证。

入口：http://192.168.8.16:43312/df-v001/iter60-calibration-r1/

工程：`/home/zjb/Desktop/worktrees/mfu-16to256`，分支`32to256`。下述相对路径以此为根。

## 已经实现

`calibrate256_fine_r1.py`复用原详细F/B构建器，显式替换物理rank组、MB选择和真实256配置日志。生成的适配器代码、每步输入/输出SHA、日志和完成标记均落盘。原始trace只读；源码冻结结果不覆盖。

`audit256_fine_binding_r1.py`把目标事实映射回共同图的节点和前驱，核对缺项、类型、非负就绪参数及每个出口。缺项不以零代替，不按整块比例缩放。原冻结图审计与修正候选审计分目录保存。

PP1 MB0第一组rank16–23：四层F 327.149789 ms，B 818.871418 ms。第二组rank24–31：F 327.883475 ms，B 825.146288 ms。共同图保留F 1106节点、B 5514节点；候选绑定出口误差均在1e-5 ms以内。实测就绪间隔参与标定，因此属于条件核账，不能宣称独立预测零误差。

## 依赖修正候选

原图的两处不相容位置是`R1:r8:s0:n5`、`R3:r13:s0:n5`；映射至物理rank16和21。256 trace显示本地V宽度拷贝在先前CP完成前已开始，原边产生约−10.501/−10.691 ms就绪差。

CPU `aten::contiguous`输入为`[2,2,4096,64,128]`，对应设备拷贝与CP活动实测重叠。`repair_recompute_vpack_r1.py`建立`iter60-shared-dag-r2`候选，对所有对应V输入拷贝采用同一规则，移除15条按源时序推定的CP完成前驱。源成本模块不变，源保留就绪参数显式重新基准化；原32卡全部节点出口保持不变。该候选不能称为与原冻结图逐边完全相同。

## 全量覆盖

覆盖清单为PP1–14 × 两组EP8 × 四MB，共112单元。每个单元含四层F与完整四层B。批量任务是本地进程，不是新agent。

最新计数见`results/data-foundation/calibrate256-fine-r1/batch-progress.json`及网页`report.json`；首批另行运行的16:1与224:0不在batch的110项中，网页按完整112项检查。

暴露的问题必须保留：

1. CP不能总按trace文件事件序号关联；`extract256_cp_identity_r2.py`通过CPU marker、External id、correlation及组/消息量字段严格关联，rank16–23 MB3的128个CP事件通过。
2. rank227 PP14 MB0第一个反向单元，两次`_GroupedLinearBackward`输入是`[0,5120]`、`[0,3072]`。CPU调用存在、GPU专家事件为空，不是可以强行补非零耗时的漏项。`zero_work_evidence_r2.py`仅为输入含零维且没有关联设备工作的CPU调用建立零工作量边界，事件ID明确为null；隔离候选尚待完整核验。
3. rank20 MB2有同device、context/stream上的两个拷贝事件重叠约0.026893 ms，事件92623、92663。未抹平时间戳，也未把负参数裁零来冒充通过。
4. 目标的就绪余量可能大于原敏感性测试固定的2 ms扰动；测试不通过不自动等于依赖错误，需要分别核对。

## 尾部

`build256_tail_stream_r1.py`生成5条实际stream、570个设备活动及显式保留间隔的图，回放376.997416 ms。Dense RS 23.797036 ms；EDP RS 55.193816 ms；两次Dense AG合计25.818284 ms、两次EDP AG合计28.932575 ms；优化器关联活动合计29.077772 ms；模型并行组两个同步/统计allreduce合计169.211871 ms。

这些是活动时长，可重叠，不可直接相加当尾部总长。图仅明确同stream顺序，跨stream及全rank就绪仍是参数，不是已验证的完整因果图。没有把实测总尾长作为额外节点再加一次。

## 命令与证据目录

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
python3 workflow/data_foundation/calibrate256_fine_r1.py --rank-base 16 --mb 0
python3 workflow/data_foundation/audit256_fine_binding_r1.py --rank-base 16 --mb 0 --model-dir results/data-foundation/iter60-shared-dag-r2 --tag shared-binding-r2
python3 workflow/data_foundation/publish256_calibration_r1.py
```

已完成构建步骤可以仅在输入/输出哈希一致时恢复；审计输出目录不能直接覆盖。新研究版本必须用独立目录。不要在批量运行过程中重复启动相同单元。

- `results/data-foundation/calibrate256-fine-r1/r{rank_base}-mb{mb}/`：每步事实、日志、适配器、完成标记与图。
- `.../frozen-binding-r1/`：原图兼容检查，保留失败。
- `.../shared-binding-r2/`：共同候选绑定与检查。
- `results/data-foundation/iter60-shared-dag-r2/`：修正规则、证据和源图回归。
- `results/data-foundation/iter60-tail-stream-r1/`：尾部参数、图、证据与SHA。
- `results/data-foundation/calibrate256-cp-identity-r2/r16-mb3/`：严格CP身份适配证据。
- `results/data-foundation/calibrate256-zero-r2/`：零工作量隔离尝试，不混入r1结果。

浏览器入口HTTP 200、112格可点击、无脚本错误。修改后的CP组选择对原32卡iter155的全部pair图与冻结值完全相等；临时回归目录`/tmp/mfu-cp-pair-regression-0hw3br0r/output`。

未提交或推送；旧发布版本和旧精度结论没有被改写。

## 最终追加 · 2026-10-23

上文保留过程状态。现112/112单元均通过，包括严格CP、零工作量、紧凑非零激活/转换序列及重叠适配，原失败日志不删除。完整图组装与独立重放已完成：372571节点，419866边；1F1B 20378.337575ms对20396.013187ms，误差−0.086662%；训练主体20755.334991ms对20773.010603ms，误差−0.085089%。MFU回放4.870326%，实测耗时推导4.866182%。

这仍是同轮实测成本与保留间隔条件回放，不是新外推精度。完整范围、图修正、证据清单和复现方法见ITER60_FINE_MODEL_R1.md；最终网页 `/df-v001/iter60-fine-model-r1/`。本轮标定目标完成，独立预测与完整因果验证另行研究。
