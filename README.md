# mfu-model · F/B 冻结分支 `16to256`

当前冻结的是 **32卡→256卡** 的CP2/EP8细化F/B与1F1B外推；分支名称沿用用户指定的`16to256`，不代表本版以16卡为源。

入口：[F/B分支交接](docs/data-foundation/FB_BRANCH_HANDOFF.md)。冻结文件及哈希见[快照清单](docs/data-foundation/FB_BRANCH_SNAPSHOT.json)。

离线验证（无需原始trace）：

```bash
python -B workflow/verify_fb_snapshot.py --out /tmp/mfu-fb-freeze-check-new.json
```

当前结果：32卡iter70区间误差2.80%；256卡iter60 rank0首F→末B区间预测偏大6.16%。不含更新尾段，不是全世界MFU精度；模型仍保留stage级PP、EP8副本同成本及源就绪残余假设。两套展开图/回放可由已提交模板逐字节重建。

## 继承的历史 v6.10 基线

基于计算图的训练迭代时间与 MFU 建模工程。原分支 `feat/w37-v610` 接入并复现
W37 v6.10：使用源 256 卡派生观测及封存参数，在目标 224 卡、PP14、3 microbatch
计算图上回放、评价和展示完整迭代。

**当前状态：冻结复现 PASS，不是新的盲测精度结论。**
源版本为 `b8330a0`，清单审计为 `146cc97`。接入没有修改模型参数、拓扑或校准窗口。
旧 v2/v4 的约 0.1% 误差属于 256 卡同配置校准回放，不能作为 v610 外推精度。

## 文档与页面入口

- [v610 完整复现说明](docs/v610/README.md)：环境、数据、命令、参数接口和边界。
- [验收报告](docs/v610/ACCEPTANCE.md)：冻结结果、测试及已知限制。
- [机读运行清单](docs/v610/run_manifest.json)：代码、配置、输入、结果和日志的路径与 SHA。
- [三阶段与计算图模型入门](docs/MFU_THREE_STAGE_AND_DAG_MODEL_GUIDE.md)：基础概念与历史建模说明。
- [历史 v1–v4 README](docs/LEGACY_V1_V4_README.md)：原校准报告、OISA 接入及旧运行方式。

局域网页面（需要本机服务运行，非公网部署）：

- [当前 v610 报告](http://192.168.0.49:43311/w37-report.html)：完整 iter、参数列表和四轮对比。
- [独立参数图册](http://192.168.0.49:43311/v610-parameter-atlas.html)：整轮 → F/B模块 → 参数与来源。
- [历史校准报告与图册](http://192.168.0.49:43311/w37-report-history.html#calibration)。

## 模型与冻结结果

完整图保留 **327746 个节点、364784 条边**，含零成本节点；
细粒度源参数共 **6765 条**，实际绑定 **4387 个键、70192 个唯一更新节点**。

节点成本分别保存计算曝露、计算重叠、网络传输、软件同步和框架余项；
依赖等待与关键路径由图传播体现，不能把各节点耗时或网络 FCT 直接相加当作迭代时间。
继承的通信、优化器、形状倍率与旧时长下限均保留。

| 冻结指标 | v610 复现值 |
| --- | ---: |
| 图内 raw 时间 | 20953.110588 ms |
| ProfilerStep | 21557.550256125 ms |
| Training Step | 22930.003646124995 ms |
| MFU（由 Training Step 派生） | 3.285055047379364% |

图内 entry 为 1265.388210 ms；图外 raw→Profiler 补差为 604.439668124998 ms，
outer 为 1372.453390 ms，不能重复计时。MFU 使用冻结有效 FLOPs
`8.436548311982576e16 / iter`、224 卡、单卡峰值 `500e12 FLOP/s`。

目标 85/90/95/100 四轮是**已暴露开发评价**，Profiler / Training MAPE 分别为
9.6493% / 10.0438%，不是同配置回放约 0.1% 的那组结果。

验收已完成：26 张科学表解压后字节一致；40 项回归检查、32 项测试、
84 项当前页面检查、17 条链接和 23 项历史页面检查通过。详见验收报告。

## 快速复现

前提：Linux、`bwrap`、`uv`，以及单独取得的冻结派生制品包。
**仅克隆 Git 仓库不足以运行：原始 trace、大型派生输入和完整生成结果不随 Git 分发。**
制品包按 [artifact_manifest.json](docs/v610/artifact_manifest.json) 校验，不得修改旧 SHA 绕过不一致。

在仓库根目录执行；已有 `.venv` 时跳过创建，结果目录必须使用新名称：

```bash
uv venv --python 3.11.15 .venv
uv pip install --python .venv/bin/python -r requirements-v610.lock.txt
uv pip install --python .venv/bin/python --no-deps -e .

# 本机制品位置；其他机器请改为实际存放目录。
export MFU_V610_ARTIFACT_ROOT=/home/zjb/Desktop/mfu-model-artifacts/w37-v610-b8330a0

# 发布管线：回放封存参数 → 评价 → HTML。
.venv/bin/python -m snakemake --snakefile workflow/v610/Snakefile --cores 1 \
  --config pipeline=release run_root="$PWD/results/v610/release-next"

# binding 管线：源派生观测拟合 → 四组消融 → 审查 → 评价。
.venv/bin/python -m snakemake --snakefile workflow/v610/Snakefile --cores 1 \
  --config pipeline=binding run_root="$PWD/results/v610/binding-next"

.venv/bin/python -B scripts/v610/verify.py \
  --release results/v610/release-next --binding results/v610/binding-next \
  --output results/v610/verification-next.json
.venv/bin/python -m pytest -q
```

两条管线单线程运行；输出仅写本工作区 `results/v610/`，不覆盖非空 stage。
不需要原 W37 源码或旧虚拟环境，不运行原始 trace 全量重扫、旧全局 Snakemake 或 OISA 仿真。

生成完整阅读版（八章结构及独立 v610 参数图册），再启动服务；只渲染封存数据，不重跑模型：

```bash
.venv/bin/python -B scripts/v610/build_reading_ui.py \
  --run-root results/v610/release-next --output results/v610/ui-next
.venv/bin/python -B scripts/v610/serve.py \
  --run-root results/v610/release-next --ui-root results/v610/ui-next \
  --host 127.0.0.1 --port 0
```

终端会显示实际访问地址；`--port 0` 自动选空闲端口。需要局域网访问时改为
`--host 0.0.0.0`。本机已有的验收结果目录为 `results/v610/release-final` 和 `binding-final`。
当前展示修订及复现方法见 [UI_READING_REVISION.md](docs/v610/UI_READING_REVISION.md)；冻结原报告仍保留。

## 工程入口与数据边界

| 路径 | 用途 |
| --- | --- |
| `configs/v610/` | 版本登记、冻结 MFU 常量、路径配置样例 |
| `src/x10000_analysis/v610.py` | 输入映射、SHA 校验、只读隔离执行、新 seal |
| `vendor/fab_w37/` | 原样保留的算法、配置、校准历史及精简证据 |
| `workflow/v610/Snakefile` | release / binding 显式管线 |
| `workflow/profiler/Snakefile` | profiler 派生输入主线；通信日志诊断可选 |
| `docs/v610/` | 交接、清单、参数边界与已纳管验收证据 |
| `results/v610/` | 本地新生成结果与日志，Git 忽略 |

源细粒度拟合使用 85/90；源 95/100 仅用于增量开发分析。模型阶段禁止解释性读取目标
evaluator 输入，且保留全部既有通信成本，不因额外日志变可选而删除历史成本项。

原始数据未搬迁、未入 Git；未来搬迁须保留旧位置软链接并验证原 SHA。
完整源兼容图回放、继承时长下限的独立物理归属，以及 raw→所有历史校准项的生成链
仍未完成验证。16→256 的 source-only 研究不能隐含使用本版 256 卡时序校准参数。
