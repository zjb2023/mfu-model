# B最后执行层：60/70波动证据审计

范围：32卡PP1 B0，rank8–15，本stage第1层（反向最后执行的unit3）。只读提取16份已有trace，不调参，不改变DAG或历史预测。

## 事实与解释边界

1. 每卡20个本地专家。使用重计算前向GroupedLinear的tokens_per_expert，并用Sequence number将FC2/FC1反向与前向一一对应。八卡token总数变化与两段反向耗时变化方向全部一致，但尚未建立可外推的线性函数。这里的token是专家输入行数，不是全局唯一token数。
   - rank8：37258→30830（−17.25%），FC2/FC1分别−0.092/−0.460ms。
   - rank11：35527→56319（+58.52%），FC2/FC1分别+1.593/+3.143ms。
   - rank14：25605→52120（+103.55%），FC2/FC1分别+1.719/+3.664ms。
2. Combine八卡平均跨流包络+3.828646ms，其中活动并集仅+0.019190ms、未覆盖部分+3.809456ms；通信流可见尾部+0.019275ms。不能将包络增量当作网络service增量。未覆盖只针对这些关联事件，不能声称GPU完全空闲。
3. 最晚本地准备卡60为rank8，70为rank11；最晚准备→最早可见后处理的有效残余1.253501→1.395757ms。按本层八卡最早开始对齐，最晚准备111.026550→121.946257ms。与“专家负载改变→各卡就绪时刻改变→Combine耦合延后”相符，但跨卡时钟可比性和完成耦合仍有条件假设，不是硬barrier或纯等待的证明。
4. 重计算+4.105332ms的互斥时间分解：FC1 +0.708309、FC2 +0.350095、其他关联活动+0.730334、无关联活动覆盖+2.608508、跨分类重叠+0.540982、CP −0.595611、Dispatch −0.263480、Combine +0.026195ms。各项按活动集合互斥划分，可相加；不能将未覆盖归因于某一个具体机制。

## 参数含义

- FC1/FC2：每专家token分布、矩阵尺寸是工作量输入候选；本次target token仅诊断，不注入60→70预测。
- Combine：保留本地准备、组内就绪和可见后处理分项，不能用整个跨流包络充当纯通信成本。
- 重计算：当前仍保留源包络；本次分解是定位依据，不额外叠加成本。

用户决策（2026-09-20）：token工作量驱动的FC1/FC2成本候选记为 `DEFERRED_USER_DECISION`，仅保留证据与接口设想，暂不拟合、不实施，不作为32→256外推的前置条件。

若以后恢复：以60作为唯一校准数据建立专家GroupedLinear的少参数工作量成本候选；先声明70 token为“已知工作量条件输入”的实验，再与不提供目标token的源参数基线分开比较，禁止混称source-only。是否需要未来token预测器是另一项任务。

当前方向：在对应局部张量规格、CP/EP配置一致的前提下，先复用32卡现有计算/通信有效成本，推进中间PP stage扩展。配置一致不等于路由token分布或运行耗时逐次一致；本次波动证据保留为迁移假设的限制，不据此增加调参任务。

## 版本、复现与验证

工作区 `/home/zjb/Desktop/worktrees/mfu-w37-v610`。

```bash
python -B workflow/data_foundation/audit_b_last_layer_variation_r1.py --out /tmp/b-last-layer-variation-new
python -B workflow/data_foundation/publish_b_last_layer_audit_r1.py
node workflow/data_foundation/test_b_last_layer_audit_r1.cjs results/data-foundation/b-last-layer-variation-ui-check
```

输出必须不存在；发布器固定从已保存r1结果生成新的r3页面，重复发布不覆盖。
`results/data-foundation/b-last-layer-variation-r1/`：evidence-60/70保存原trace路径、SHA、CPU/GPU索引、每专家token向量、分流时间；report/manifest保存统计与代码/输出哈希。
验证：同口径重计算与前次评价一致；八卡FC1/FC2与token变化方向一致；包络=活动并集+未覆盖，重计算互斥分解求和正确，代码/输出哈希检查通过。

HTML：http://192.168.8.16:43312/df-v001/b-r10-60to70-r3/#last-layer-audit

旧r2页面保留。新页面保留原预测/实测图，添加工作量、Combine分解、八卡就绪、重计算分解四张表。没有新增预测精度结论，没有提交或推送Git。

浏览器检查PASS：HTTP200、四张审计表、原八卡切换功能、390px无整页溢出、无JS错误。证据 `results/data-foundation/b-last-layer-variation-ui-check/checks.json` 与已检查的 `audit.png`。
