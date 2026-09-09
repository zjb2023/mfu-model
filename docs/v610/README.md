# mfu-model v6.10 冻结复现分支

本地分支 `feat/w37-v610`，工作区 `/home/zjb/Desktop/worktrees/mfu-w37-v610`。
从 mfu `e71ccb0` 独立接入 W37 `b8330a0`（审计 `146cc97`），未合并、未改 W37 源材料。
冻结接入先完成本地提交；随后按用户单独授权，分支已推送至 `origin/feat/w37-v610`（冻结验收提交 `3b1a2b6`）。
验收结果见 [ACCEPTANCE.md](ACCEPTANCE.md)；机器证据见 [acceptance/verification.json](acceptance/verification.json)。

## 当前版本和目录

- `configs/v610/version.json`：当前版本登记；`physics.json`：保持原值的 MFU 常量镜像与物理口径限制。
- `vendor/fab_w37/`：426 个原样代码、配置、文档/精简证据文件；SHA 在 `imported_code.json`。其中 388 个核对了源 Git blob，38 个历史结果此前仅有审计 SHA、未在源 Git 受控；明细见 `provider_git_provenance.json`。本分支将其明确纳管，不倒推为源提交已有。只作为冻结快照，不从此目录直接运行历史脚本的默认路径。
- `src/x10000_analysis/v610.py`：仅重定位输入、隔离执行、登记新 seal；没有改成本、拓扑、校准窗口或公式。
- `workflow/v610/Snakefile`：显式 release/binding 两条管线。release 只回放封存参数；binding 从源派生观测重新拟合并运行原四组消融。
- `workflow/profiler/Snakefile`：默认只校验源 profiler 派生观测；通信日志不再是此默认入口的前置依赖。
- `docs/v610/artifact_manifest.json`：544 条原始审计记录、逻辑→物理映射、371 个原审计对象及 8 个补充网页对象（合计 379）。补充项另记在 `supplement_manifest.json`，没有冒充旧 seal。
- `results/v610/`：本 worktree 的新结果与执行日志（Git 忽略）；历史大图、数据与旧结果留在外部制品包。

外部派生/报告包：`/home/zjb/Desktop/mfu-model-artifacts/w37-v610-b8330a0`。
包内均为独立普通文件副本，不是指向 W37 的软链接/硬链接。原始 trace 没有读取内容、复制、移动或纳入 Git。
这不是纯 Git 自包含数据集：重新克隆必须取得该制品包，按原 SHA 校验；源码和 Python 入口无需原 W37 目录或旧虚拟环境。

## 最小复现

在本工作区执行。需要 Linux `bwrap`（本机已有），Python 3.11.15；CPU 单线程，无 GPU、原始 trace 重扫或 OISA 仿真。
新建环境时执行；本机已完成，不要覆盖已有 `.venv`：

```bash
uv venv --python 3.11.15 .venv
uv pip install --python .venv/bin/python -r requirements-v610.lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
export MFU_V610_ARTIFACT_ROOT=/home/zjb/Desktop/mfu-model-artifacts/w37-v610-b8330a0
```

也可使用本机已存在的 `.local/v610.json`（忽略入 Git），样例在 `configs/v610/paths.example.json`。
每次重跑必须使用新的结果目录；非空 stage 会拒绝覆盖。最终验收目录是 `release-final`、`binding-final`（早期成功运行 `binding-reproduction-r1` 也保留）。

```bash
.venv/bin/python -m snakemake --snakefile workflow/v610/Snakefile --cores 1 --config pipeline=release run_root="$PWD/results/v610/release-next"
.venv/bin/python -m snakemake --snakefile workflow/v610/Snakefile --cores 1 --config pipeline=binding run_root="$PWD/results/v610/binding-next"
.venv/bin/python -B scripts/v610/verify.py --release results/v610/release-next --binding results/v610/binding-next --output results/v610/verification-next.json
.venv/bin/python -m pytest -q
```

完整 stage SHA、解释性数据读取和命令分别见各 stage 的 `complete.json`、`adapter_access.json`、`command.json`。
启动前可以对所有声明输入做字节 SHA 校验；模型阶段禁止解释性读取 evaluator 数据。原 release/binding 目标读取探针仍会被拒绝。
每个 worker 内 W37、fabric-data-analysis 和原 mfu 目录被隐藏，网络被隔离，整个文件系统只读，仅当前输出 stage 可写。

## 参数和数据接口

| 项目 | 冻结语义 / 入口 |
| --- | --- |
| 图 | `model/nodes.csv.gz` 327746 行，`edges.csv.gz` 364784 行；node_id / 依赖语义和拓扑 SHA 不变 |
| 成本 | `duration_ns` 等于 compute_exposed、compute_overlap、network_service、software_sync、framework_residual 五类 `_ns_model` 之和；包含零成本节点 |
| 时序 / 关键路径 | predicted_start_ns、predicted_end_ns、critical_predecessor、on_critical_path；ns，不以终点相同代替逐节点验收 |
| 细粒度参数 | `fine_parameters.csv` 6765 条；`node_parameter_bindings.csv.gz` 4387 键、70192 唯一节点、224 rank / PP14 / MB3 |
| 继承成本 | 原 58 聚合键、PP 前后向、CP/EP/DP/EDP、RS/AG、优化器、软件/框架项和形状倍率均保留；不只保留被更新节点 |
| 曝露成本下限 | 曝露节点 `max(old_duration,new_compute)`，overlap 节点替换独立计算时长；不把继承 floor 宣称为已独立验证的物理模型 |
| 阶段账本 | entry 1265.388210 ms 已在图中；raw→Profiler 补差 604.439668124998 ms、outer 1372.453390 ms 在图外，不重复计算 |
| MFU | 有效 FLOPs 8.436548311982576e16 / iter，224 卡、单卡 500e12 FLOP/s；由 Training Step 推导，不改冻结常量 |

源 256 细粒度拟合窗口为 85/90；95/100 只是源增量开发分析，不是完整兼容源图独立验证。
目标 224 的 85/90/95/100 是已暴露开发评价，曾参与版本选择；迁移 PASS 不是新的精度或盲测结论。
16→256 研究不能读取这些 256 时序校准参数作为隐含 source-only 输入。224 分支 FLOPs/外推模块已审阅，但没有整仓合并或引入会改变结果的策略变换。

## Profiler 主线、可选日志与旧 workflow

```bash
.venv/bin/python -m snakemake --snakefile workflow/profiler/Snakefile --cores 1
```

该默认任务只 SHA 校验已封存源观测表，不需要 MTLINK/NIC/额外 DeepEP 日志，也不扫描 raw。
原分析 workflow、case256 分析脚本和通信成本来源完整保存在 vendor；**不要执行 vendor 的全局 Snakemake**，其旧默认路径属于历史来源。
新增入口仅提供元数据清点边界，不声称已完成 raw→全部历史成本生成链。

未来获得单独 raw 清点授权时，显式配置 `framework_root` 并指定 `profiler_inventory` 规则；该规则只验证文件网格，256 rank/32 host，不解析 trace 语义。
`optional_communication_inventory` 只有显式配置 `communication_root` 和已有 `rank_host_map` 才存在，按需读取硬件 CSV；当前未对真实日志执行。
DeepEP 等后续历史诊断脚本保留为可选代码，不会被默认 DAG 调度。改变成本来源或新增 profiler-only 模型须另起模型版本，不能覆盖 v6.10。

## 数据搬迁契约

没有指定新 raw 存储目的地，所以本轮不移动原始数据。未来搬迁需逐个明确目标、迁移后在旧位置留软链接、记录 resolve 后真实路径和原 SHA；断链或 SHA 不符必须停止，不改旧 SHA。
该契约已用小型临时文件验证；这不等于真实 raw 已搬迁。制品包也可整体迁移后设置 `MFU_V610_ARTIFACT_ROOT`，历史逻辑路径和 seal 不改。

## 页面与离线阅读

本机独立服务目前为 `http://192.168.0.49:43311/research.html`，原 8037 服务不变。
新服务首页指向 v6.10；历史页 `/w37-report-history.html#calibration` 保留完整旧校准史，旧图册 / 教学 DAG 能往返。历史 v685 参数图册不替代当前 v610 的 6765 参数说明。
原始 HTML/payload/SHA 保留，服务仅在响应中重写旧链接，并提供 UTF-8 Markdown 页面；新重生成内容也有独立 SHA。

```bash
.venv/bin/python -B scripts/v610/serve.py --run-root results/v610/release-final --host 0.0.0.0 --port 0
npm ci
node scripts/v610/check_browser.cjs release
node scripts/v610/check_browser.cjs history
```

`--port 0` 自动找空闲端口，写入 `results/v610/service.json`。不要重复占用已运行的 43311；服务退出后可用此命令重新启动。
浏览器测试使用本机 Chromium 1228 缓存；缺失时 `npx playwright install chromium`，不依赖旧工程的 node_modules。已验收下载后的当前单文件 HTML 和历史内嵌报告离线能力；外部证据下载仍需要完整制品包。

## 已知限制和下一步

全量 raw→全部历史校准项仍未闭环；完整源兼容图回放、继承 floor 的独立物理归属未验证；raw 尚未搬迁。仅迁移复现与本地版本管理完成，不借此宣称模型精度改善。
本次到本地提交结束。下一次研究若授权，优先单独验证源完整兼容回放；不要直接继续调参或启动另一任务。
