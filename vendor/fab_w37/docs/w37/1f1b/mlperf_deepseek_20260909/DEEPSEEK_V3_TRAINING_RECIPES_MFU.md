# DeepSeek-V3：MLPerf 训练配方、Step 时间与 MFU 估算

整理日期：2026-09-09。范围：MLPerf Training v6.0，NVIDIA 提交的 DeepSeek-V3 671B 配方。

本文使用普通 Markdown 表格和纯文本公式，不使用 LaTeX。

## 1. 结果总览

**256 张 GB300、512 张 GB300、8192 张 GB200 的 MFU 估算分别为 32.48%、31.73%、11.97%。这些数值是按公开配置和日志推算的，不是 MLPerf 官方报告的 MFU。**

所有配方的模型均为 DeepSeek-V3 671B，任务为 MoE 预训练基准，训练框架为 PyTorch / NeMo / Megatron-Bridge，精度路径为 MXFP8，序列长度为 4096。已匹配的系统文件声明 NeMo Framework Release 26.04；R10 未匹配该规模的系统文件，不能仅凭代码所在目录确认实际运行版本。

| 配方 | 模型 | 硬件 | Host 数 × GPU/Host | GPU 总数 | GBS | MBS | 平均 Step 时间（秒） | 模型 TFLOPS/GPU | 估算 MFU |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| R10 | DeepSeek-V3 671B | GB200 | 64 × 4 | 256 | 15360 | 1 | 未匹配日志 | — | — |
| R12 | DeepSeek-V3 671B | GB300 | 64 × 4 | 256 | 15360 | 1 | 39.458 | 1624.0 | **32.48%** |
| R11 | DeepSeek-V3 671B | GB300 | 128 × 4 | 512 | 15360 | 1 | 20.196 | 1586.4 | **31.73%** |
| R13 | DeepSeek-V3 671B | GB200 | 2048 × 4 | 8192 | 16384 | 1 | 3.569 | 598.5 | **11.97%** |

R10 保留为公开配置配方；没有用其他 GPU 型号或其他规模的耗时填补缺失数据。GB300 对应日志的 `submission_platform` 文本沿用了 GB200 名称，本文硬件型号依据配置脚本与对应系统 JSON 交叉确认。

## 2. 并行策略

| 配方 | TP | PP | CP | EP | DP（batch 口径） | ETP | SP | VPP | 调度与通信 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| R10 | 2 | 4 | 1 | 32 | 32 | 1 | 开启 | 4 | Interleaved PP；自定义流水线层划分；MoE 通信重叠 |
| R12 | 1 | 4 | 1 | 32 | 64 | 1 | 关闭 | 4 | Interleaved PP；自定义流水线层划分；MoE 通信重叠 |
| R11 | 1 | 4 | 1 | 32 | 128 | 1 | 关闭 | 4 | Interleaved PP；自定义流水线层划分；MoE 通信重叠 |
| R13 | 2 | 4 | 1 | 32 | 1024 | 1 | 开启 | 4 | Interleaved PP；自定义流水线层划分；MoE 通信重叠 |

- **TP**：张量并行；**PP**：流水线并行；**CP**：上下文并行；**EP**：专家并行。
- **DP**：用于计算全局 batch 的稠密数据并行副本数，不代表专家参数的数据并行组大小。
- **ETP**：专家张量并行；**SP**：sequence parallel，不等于 CP。
- **VPP**：每个物理流水线 stage 的虚拟分块数，不额外增加 GPU 数。
- 这些 MoE 配方不能把 TP、PP、CP、EP、DP 当作五个独立维度相乘；EP 不应在下式中再次除掉。专家通信组按框架自身的映射构建。

```text
DP = GPU 总数 / (TP × PP × CP)

R10:  256 / (2 × 4 × 1) =   32
R12:  256 / (1 × 4 × 1) =   64
R11:  512 / (1 × 4 × 1) =  128
R13: 8192 / (2 × 4 × 1) = 1024
```

自定义流水线布局及重计算、通信重叠开关应保留原始配置中的定义，不能仅凭 PP=4 假设四段计算量完全相等。

## 3. 训练参数

| 配方 | GBS | MBS | MINIBS | GA：每次更新的微批次数 | 序列长度 | 每次更新的全局 Token 数 | 精度 |
|---|---:|---:|---:|---:|---:|---:|---|
| R10 | 15360 | 1 | 480 | 480 | 4096 | 62,914,560 | MXFP8 |
| R12 | 15360 | 1 | 240 | 240 | 4096 | 62,914,560 | MXFP8 |
| R11 | 15360 | 1 | 120 | 120 | 4096 | 62,914,560 | MXFP8 |
| R13 | 16384 | 1 | 16 | 16 | 4096 | 67,108,864 | MXFP8 |

- **GBS**：一次优化器更新覆盖的全局序列数。
- **MBS**：每个 DP 副本每个 microbatch 的序列数。
- **MINIBS**：该提交脚本中，每个 DP 副本一次更新处理的序列数，包含梯度累积，不是 MBS。
- **GA**：一次优化器更新中，每个 DP 副本处理的 microbatch 数。
- 已匹配的 R12、R11、R13 日志报告优化器为 AdamW；分布式优化器及 overlap 的具体设置见各配方参数源。R10 不标为日志确认。
- MXFP8 描述低精度计算路径，不表示所有梯度、优化器状态和运算都使用 FP8。

```text
MINIBS = MBS × GA
GBS = MBS × DP × GA = MINIBS × DP
每次更新的全局 Token 数 = GBS × 序列长度

R10: 1 ×   32 × 480 = 15360
R12: 1 ×   64 × 240 = 15360
R11: 1 ×  128 × 120 = 15360
R13: 1 × 1024 ×  16 = 16384
```

此处是 MLPerf 性能基准配方，不能将基准运行时间理解为从随机初始化完成模型全部预训练的时间。

## 4. 平均 Step 时间的取样方法

| 配方 | 日志 | 计入的 Step 数 | 平均值（秒） | 中位数（秒） |
|---|---|---:|---:|---:|
| R10 | 未匹配 | — | — | — |
| R12 | result_0.txt | 50 | 39.457955 | 39.211013 |
| R11 | result_0.txt | 50 | 20.196413 | 20.092838 |
| R13 | result_0.txt | 45 | 3.568921 | 3.494354 |

统计规则：

1. 读取每个对应提交的 `result_0.txt`，解析 `:::MLLOG` JSON 记录。
2. 只选 `key=tracked_stats` 且 value 同时包含 `train_step_time` 和 `reduced_train_loss` 的记录，对应 `DeltaTimingCallback`。
3. 不把另一套只记录 `train_step_time` 的回调再次计入，避免同一步重复统计。
4. 对全部选中的训练 Step 求算术平均，没有额外裁剪慢 Step 或人为选择最快区间。
5. 此回调在验证结束时重置计时，避免把验证耗时累加到验证后的第一个训练 Step。

**这里的 Step 是一次训练更新，不是一个 microbatch。** 它是提交日志记录的训练步骤耗时，包含框架与分布式执行影响；不是单个 GPU kernel 耗时，也不是单个 PP stage 的纯计算时间。

只使用每个提交的一个 `result_0` 运行，未对所有重复运行求平均，未计算跨运行置信区间。没有使用 time-to-convergence 直接代替训练 Step 时间。

## 5. 基于硬件峰值估算 MFU

### 5.1 分子：模型计算量

沿用提交代码 `config_common.sh` 的计算量约定：

```text
MODEL_TFLOP_PER_SAMPLE = 1068
F_sample = 1068 TFLOP / sequence
F_step = 1068 × GBS                  单位：TFLOP / step
```

这里的一个 sample 是该配方中长度 4096 的序列，不能再额外乘一次 4096。它也不等于简单地让全部 671B 参数在每个 Token 上执行一次；MoE 只激活部分专家。

此报告没有独立逐算子复核 1068 TFLOP 的推导、因果 Attention FLOPs 约定或重计算计入方式，因此结果应称为“沿用提交 FLOPs 口径、按 FP8 峰值归一化的 MFU 估算”。

### 5.2 分母：硬件非稀疏 FP8 标称峰值

| GPU | NVL72 每柜 GPU 数 | 每柜 FP8 稀疏峰值 | 每 GPU 非稀疏 FP8 峰值（本报告口径） |
|---|---:|---:|---:|
| GB200 | 72 | 720 PFLOPS | 5 PFLOPS = 5000 TFLOPS |
| GB300 | 72 | 720 PFLOPS | 5 PFLOPS = 5000 TFLOPS |

```text
每 GPU 非稀疏 FP8 峰值 = 720 / 72 / 2 = 5 PFLOPS
```

NVIDIA 规格页将 Tensor Core 指标标为包含稀疏加速，除非特别注明。这里使用非稀疏峰值：MoE 的专家选择不等于 Tensor Core 的 2:4 结构化稀疏加速。该分母是标称峰值，并未按提交运行中的实际 GPU 时钟、功耗上限进行修正；也没有把 FP8、BF16 混合计算拆成各自的理论时间。

### 5.3 纯文本公式及逐项计算

```text
N = GPU 总数
T = 平均训练 Step 时间，单位秒
P = 单 GPU 非稀疏 FP8 标称峰值，取 5000 TFLOPS

模型 TFLOPS/GPU = (1068 × GBS) / (N × T)
估算 MFU (%) = [(1068 × GBS) / (N × T × 5000)] × 100
```

```text
R12：256 张 GB300
模型 TFLOPS/GPU = (1068 × 15360) / (256 × 39.457955499)
                  ≈ 1624.01
估算 MFU         = 1624.01 / 5000 × 100
                  ≈ 32.48%
```

```text
R11：512 张 GB300
模型 TFLOPS/GPU = (1068 × 15360) / (512 × 20.196413360)
                  ≈ 1586.42
估算 MFU         = 1586.42 / 5000 × 100
                  ≈ 31.73%
```

```text
R13：8192 张 GB200
模型 TFLOPS/GPU = (1068 × 16384) / (8192 × 3.568921359)
                  ≈ 598.50
估算 MFU         = 598.50 / 5000 × 100
                  ≈ 11.97%
```

计算使用未四舍五入的平均时间，显示值为四舍五入结果。R10 缺少匹配的 Step 时间，无法用这套方法估算 MFU。

## 6. 用于 SimAI / AICB 对照时的含义

- 优先将硬件、训练框架、并行策略和 batch 参数作为同一组配方，再对照平均 Step 时间；MFU 是派生指标。
- 本文 MFU 是整个分布式训练的模型效率，包含通信、流水线等待等影响，不能直接填成单个 PP stage 的计算效率，否则可能在仿真中重复计算通信影响。
- 8192 卡的单 Step 更短，但单位 GPU 的模型 FLOPS 利用率更低；仅凭这些日志不能把损失全部归因于网络，还需分解流水线气泡、通信、路由负载和计算开销。
- 比较其他报告的 MFU 前，应统一精度峰值、稀疏口径、模型 FLOPs 公式、统计范围以及是否包含重计算。

## 7. 每条配方的原始来源

所有 GitHub 链接固定到提交仓库 commit `eabf23a07b2a0c60a289ff871dc3a46fff0d0421`。

### R10：256 张 GB200

- [训练配方脚本](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/deepseekv3_671b/implementations/theia_ngpu512_ngc26.04_nemo/config_GB200_64x4x480xtp2pp4ep32cp1_mxfp8_full_cg.sh)
- [模型及并行参数 YAML](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/deepseekv3_671b/implementations/theia_ngpu512_ngc26.04_nemo/conf/model/671b.yaml)
- [训练入口](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/deepseekv3_671b/implementations/theia_ngpu512_ngc26.04_nemo/pretrain.py)
- [容器构建文件](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/deepseekv3_671b/implementations/theia_ngpu512_ngc26.04_nemo/Dockerfile)
- 未匹配该规模训练日志和系统文件；只有配置信息。

### R12：256 张 GB300

- [训练配方脚本](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/deepseekv3_671b/implementations/theia_ngpu512_ngc26.04_nemo/config_GB300_64x4x240xtp1pp4ep32cp1_mxfp8_full_cg.sh)
- [模型及并行参数 YAML](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/deepseekv3_671b/implementations/theia_ngpu512_ngc26.04_nemo/conf/model/671b.yaml)
- [训练入口](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/deepseekv3_671b/implementations/theia_ngpu512_ngc26.04_nemo/pretrain.py)
- [容器构建文件](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/deepseekv3_671b/implementations/theia_ngpu512_ngc26.04_nemo/Dockerfile)
- [训练日志 result_0](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/results/theia_ngpu256_ngc26.04_nemo/deepseekv3_671b/result_0.txt)
- [系统硬件和框架版本](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/systems/theia_ngpu256_ngc26.04_nemo.json)

### R11：512 张 GB300

- [训练配方脚本](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/deepseekv3_671b/implementations/theia_ngpu512_ngc26.04_nemo/config_GB300_128x4x120xtp1pp4ep32cp1_mxfp8_full_cg.sh)
- [模型及并行参数 YAML](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/deepseekv3_671b/implementations/theia_ngpu512_ngc26.04_nemo/conf/model/671b.yaml)
- [训练入口](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/deepseekv3_671b/implementations/theia_ngpu512_ngc26.04_nemo/pretrain.py)
- [容器构建文件](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/deepseekv3_671b/implementations/theia_ngpu512_ngc26.04_nemo/Dockerfile)
- [训练日志 result_0](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/results/theia_ngpu512_ngc26.04_nemo/deepseekv3_671b/result_0.txt)
- [系统硬件和框架版本](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/systems/theia_ngpu512_ngc26.04_nemo.json)

### R13：8192 张 GB200

- [训练配方脚本](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/deepseekv3_671b/implementations/tyche-hsg_ngpu8192_ngc26.04_nemo/config_GB200_2048x4x16xtp2pp4ep32cp1_mxfp8_full_cg.sh)
- [模型及并行参数 YAML](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/deepseekv3_671b/implementations/tyche-hsg_ngpu8192_ngc26.04_nemo/conf/model/671b.yaml)
- [训练入口](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/deepseekv3_671b/implementations/tyche-hsg_ngpu8192_ngc26.04_nemo/pretrain.py)
- [容器构建文件](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/deepseekv3_671b/implementations/tyche-hsg_ngpu8192_ngc26.04_nemo/Dockerfile)
- [训练日志 result_0](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/results/tyche-hsg_ngpu8192_ngc26.04_nemo/deepseekv3_671b/result_0.txt)
- [系统硬件和框架版本](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/systems/tyche-hsg_ngpu8192_ngc26.04_nemo.json)

### FLOPs、计时实现与硬件规格

- [每样本 1068 TFLOP 的配置来源](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/deepseekv3_671b/implementations/theia_ngpu512_ngc26.04_nemo/config_common.sh)。
- [DeltaTimingCallback 计时实现](https://github.com/mlcommons/training_results_v6.0/blob/eabf23a07b2a0c60a289ff871dc3a46fff0d0421/NVIDIA/benchmarks/deepseekv3_671b/implementations/theia_ngpu512_ngc26.04_nemo/callback_logging.py)。
- [NVIDIA GB200 NVL72 规格](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)。
- [NVIDIA GB300 NVL72 规格](https://www.nvidia.com/en-us/data-center/gb300-nvl72/)。

## 8. 本地文件与复算

- [原始配方列表](recipes.csv)
- [估算数据及未舍入结果](deepseek_v3_mfu.json)
- [MFU 复算脚本](deepseek_v3_mfu.py)

```bash
python /home/zjb/SimAI/docs/mlperf_training_recipes/deepseek_v3_mfu.py
```

脚本只解析公开日志并执行算术计算，不启动模型训练。若临时缓存不存在，会从固定 commit 下载三份 result_0 日志。
