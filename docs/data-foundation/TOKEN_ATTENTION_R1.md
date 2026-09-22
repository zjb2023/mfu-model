# FC替换后：token重排、激活及Attention子集归因

## 结果与分母

2026-09-22。保持源32卡预测及此前代表rank诊断边界；FC固定替换为目标实测后，1F1B为20822.565050ms，对实测20396.013187ms仍高估426.551863ms（2.09135%）。对其余成本8组、256种组合重新调度，计算条件Shapley贡献。

| 已识别类别 | 贡献ms | 占剩余426.552ms |
|---|---:|---:|
| MoE token permute/unpermute |101.511|23.80%|
| 专家激活/门控（含反向） |156.497|36.69%|
| 确认的Attention非CP活动与旋转位置编码 |29.952|7.02%|
| 确认的router gating/topk子集 |0.032|0.01%|
| 尚未确认模块身份的线性活动 |65.462|15.35%|
| 尚未确认模块身份的拷贝活动 |24.566|5.76%|
| 其他未分类活动（净抵消） |-49.567|-11.62%|
| CP/EP、原重叠、间隔及组偏移的背景合计 |-54.574|-12.79%|
| 中间B其余成本净贡献 |273.879|64.21%|
| 中间B以外残余（本轮未细拆） |152.673|35.79%|

token重排＋专家激活合计258.008ms，占剩余误差60.49%；若分母改为中间B的1104.672ms，则约23.36%。两种分母不可混用。上述贡献是在FC已替换条件下分配；不能与上一版无条件75.92%直接叠加。

## 识别方法

- 复用封存的B设备事件，读取原trace核验SHA256；沿correlation找到runtime/driver，提取同线程CPU调用的完整包含关系。
- `_moe_permute_mask_map`、`_moe_unpermute_mask_map`及反向调用：明确归为MoE token重排。不能把所有Copy kernel都归入MoE。
- silu/mul及反向调用：仅当CPU发起时间位于成对专家GroupedLinear之间，归为专家激活/门控；属于时序约束推断，不是权重ID证明。
- Attention原有框架祖先和`ApplyMLARotaryEmb*`祖先：归入已确认Attention非CP子集。
- `_Linear`、`_LayerNormLinear`及反向不能仅凭名字区分Attention投影/共享专家，继续保留未确认类别。
- CP、EP祖先关联GPU、原类别间并发重叠、设备记录间隔和组起点偏移保持为背景，不再次重复分配。原other内部同时存在不同语义时保留未分类。

所有类别保持排他性时间记账，FC并集和B总成本严格守恒；基线不变。这里不是将kernel累计时长直接加到全轮。

## 限制

1. 101.511ms覆盖已确认permute/unpermute，不代表DeepEP全部打包/解包，更不代表全部通信。
2. 29.952ms不是整个Attention贡献：不含CP、未确认投影、共享重叠等；不能据此断言Attention整体只占7%。
3. 仍是源rank8、目标stage×16的有效B成本替换与外层DAG调度，非完整EP8内部节点因果验证。
4. 正负贡献均是条件诊断，不等于实际可优化收益；路由造成工作量变化的机制合理，但本轮不证明全部差异由路由导致。
5. 仅iter60，未新增跨iter验证或拟合模型；原冻结预测不改。

## 复现和证据

```bash
cd /home/zjb/Desktop/worktrees/mfu-16to256
python -B workflow/data_foundation/diagnose_token_attention_r1.py --out results/data-foundation/token-attention-r1-reproduction
```

输出目录必须不存在。有效结果 `results/data-foundation/token-attention-r1/` 包含每个stage逐设备事件的CPU祖先、分类理由、时间和原trace事件索引，以及report.json、manifest.json。输入为此前只读缓存加15个rank原始trace。

PASS：56块、256组合、输入/代码/结果哈希、旧FC替换窗口与全部中间B窗口回归、分项守恒。PARTIAL：未识别线性/拷贝的模块身份及完整通信打包解包，未验证全EP8。未提交或推送本轮新结果。
