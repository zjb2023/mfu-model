# 逐层曲线与容量证据发布 · 2026-09-23 · UI r8

页面：`http://192.168.8.16:43312/df-v001/layer-workload-r1/`。

本次提交包含224/256逐层工作量研究、256 iter60两个EP8副本的核对与合计、容量与零概率屏蔽的证据，以及③新增的收发矩阵用途和缺少的筛选前计数。所有折线连续连接；①的三个iter可独立开关。冻结预测参数不变，不包含原始trace。

关键入口：LAYER_WORKLOAD_CURVES_R1.md、LAYER_FILTERING_R1.md、LAYER_EDP_PAIR_R1.md、ROUTE_FILTER_ORIGIN_R1.md。这里的“差额”是推导未进入计算的分配量，不是实测容量丢弃量；零概率屏蔽可执行容量筛选结果，不能重复归因。

## 重建网页

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
/usr/bin/python3 -S workflow/data_foundation/publish_layer_workload_r1.py
/usr/bin/python3 -S workflow/data_foundation/publish_edp_pair_r1.py
```

生成目录：`results/data-foundation/2111-blocks-ui-r1/df-v001/layer-workload-r1/`。本地HTTP服务的镜像更新与Git推送是不同操作；两者分别检查。数据/原始路径由各结果manifest记录，原始文件只读引用、不入Git。

## 验证版本

当前r8使用`workflow/data_foundation/check_layer_release_r8.cjs`，结果在`results/data-foundation/layer-release-r8-checks/report.json`。历史UI测试和manifest为当时快照：它们可能要求PP断线，不应当作当前UI的回归条件；数据核对事实不受展示修改影响。

`extract_cohort14_expert_r1.py`仅作为提取器所需的digest/union_ms/csv_write辅助函数依赖纳入此次提交。16卡研究结果与其余无关工作不混入本次发布；GOAL.md中尚未提交的历史混合记录保留在工作区。
