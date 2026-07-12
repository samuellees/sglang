# PCIe Custom AllReduce 集成说明

本文说明如何把独立 PCIe custom allreduce 软件库接入 SGLang，并在 SGLang 中做预编译、policy 生成、smoke test、端到端 benchmark 和 Nsight timeline 验证。

## 目标和边界

- 目标：在 SM120 PCIe 机器上，把低延迟 custom allreduce 作为 SGLang 的一个可选 backend，用于 TP2/TP4/TP8 decode 通信。
- 默认行为：不开环境变量时，SGLang 仍走原有 allreduce 选择逻辑。
- 启用方式：通过 `SGLANG_USE_PCIE_CUSTOM_ALL_REDUCE=1` 显式打开。
- fallback：没有 policy、shape 不匹配、tensor 不连续、超过大小上限、TP size 不支持时，返回 `None`，由 SGLang 原有路径处理。
- 当前第一版集成不做 JIT policy 生成；kernel 通过 torch extension 预编译或首次加载编译，运行时只查 JSON policy。

## 分支内容

这一版 SGLang 分支只包含 PCIe custom allreduce 集成相关文件：

```text
python/sglang/srt/distributed/device_communicators/custom_all_reduce.py
python/sglang/srt/environ.py
python/sglang/srt/distributed/device_communicators/pcie_custom_all_reduce.py
python/sglang/srt/distributed/device_communicators/pcie_ar_config.py
python/sglang/srt/distributed/device_communicators/pcie_ar/symm_allreduce_ext.cu
python/sglang/srt/distributed/device_communicators/pcie_ar_configs/
scripts/pcie_allreduce/
docs/advanced_features/pcie_custom_allreduce_integration.md
```

`task-output/` 是本地测试产物，不属于干净分支。

## 从独立 allreduce repo 同步源码

独立通信库是 kernel 和算法实现的 source of truth。SGLang 中只 vendor 一份当前可用的 CUDA extension 源码和 policy。

推荐流程：

```bash
git clone <pcie-allreduce-repo> /path/to/pcie-allreduce
cd /path/to/pcie-allreduce
git checkout <validated-branch-or-tag>
```

把已验证的 kernel 源码同步到 SGLang：

```bash
cd /path/to/sglang
cp /path/to/pcie-allreduce/allreduce-latency/symm_allreduce_ext.cu \
  python/sglang/srt/distributed/device_communicators/pcie_ar/symm_allreduce_ext.cu
```

如果独立 repo 里源文件路径不同，只要保证最终 SGLang 内的文件存在即可：

```text
python/sglang/srt/distributed/device_communicators/pcie_ar/symm_allreduce_ext.cu
```

也可以不复制源码，通过环境变量直接指向外部源码：

```bash
export SGLANG_PCIE_AR_SOURCE_PATH=/path/to/pcie-allreduce/allreduce-latency/symm_allreduce_ext.cu
```

这种方式适合开发调试；提交给别人复现时建议把验证过的源码 vendor 到 SGLang 分支里。

## SGLang 侧接入点

接入逻辑很薄：

- `custom_all_reduce.py`：在 `dispatch_custom_allreduce()` 中优先检查 `SGLANG_USE_PCIE_CUSTOM_ALL_REDUCE`，打开后返回 `PcieCustomAllReduce`。
- `environ.py`：注册 PCIe AR 的环境变量。
- `pcie_custom_all_reduce.py`：SGLang runtime adapter，负责加载 torch extension、初始化 IPC handle、查 policy、执行 allreduce。
- `pcie_ar_config.py`：policy JSON 的读写、shape key、heuristic seed 生成。
- `pcie_ar_configs/`：默认 policy cache。
- `scripts/pcie_allreduce/`：预编译、policy 生成、smoke、E2E、GSM8K、Pareto benchmark 脚本。

运行时 shape key 是：

```text
(tp, hidden_size, batch, dtype)
```

其中 `batch = input.numel() // hidden_size`，`hidden_size = input.shape[-1]`。

## Policy cache

默认 policy 目录：

```text
python/sglang/srt/distributed/device_communicators/pcie_ar_configs/
```

目录结构：

```text
<device_name>/tp<TP>/h<HIDDEN>/<dtype>/policy.json
```

示例：

```text
SM120_5KPRO/tp2/h2048/bf16/policy.json
```

外部覆盖：

```bash
export SGLANG_PCIE_AR_CONFIG_DIR=/path/to/pcie_ar_configs
```

生成 seed policy：

```bash
cd /path/to/sglang
PYTHONPATH=python python scripts/pcie_allreduce/tune_pcie_allreduce.py \
  --tp 2 \
  --hidden 2048 \
  --dtype bf16 \
  --batch-grid default,fine \
  --max-bytes 8388608
```

默认 batch grid：

```text
1,2,4,8,16,32,64,128,256,512
```

fine grid：

```text
1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64
```

建议生产调优时至少覆盖：

```text
TP2: H2048, batch 1-64 fine grid
TP4: H4096, batch 1-64 fine grid
TP8: H6144, batch 1-64 fine grid
```

如果需要支持更大 batch，再补 `96,128,256,512`。超过 `SGLANG_PCIE_AR_MAX_SIZE_BYTES` 的 shape 会自动 fallback。

## 预编译 extension

首次启动 SGLang 时，torch extension 会编译 CUDA 源码。为了把编译开销移出服务 warmup，建议先预编译：

```bash
cd /path/to/sglang
PYTHONPATH=python python scripts/pcie_allreduce/prebuild_pcie_allreduce.py \
  --verbose
```

默认 build cache：

```text
${SGLANG_CACHE_DIR:-~/.cache/sglang}/pcie_allreduce/build
```

可覆盖：

```bash
export SGLANG_PCIE_AR_BUILD_DIR=/path/to/build
```

## 启用 backend

最小环境变量：

```bash
export SGLANG_USE_PCIE_CUSTOM_ALL_REDUCE=1
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0
```

推荐对比 PDL 时显式设置：

```bash
export SGLANG_PCIE_AR_ENABLE_PDL=0  # PDL off
# or
export SGLANG_PCIE_AR_ENABLE_PDL=1  # PDL sync/release on
```

常用调试变量：

```bash
export SGLANG_PCIE_AR_MAX_SIZE_BYTES=8388608
export SGLANG_PCIE_AR_MAX_BLOCKS=128
export SGLANG_PCIE_AR_VERBOSE_BUILD=1
export SGLANG_PCIE_AR_CONFIG_DIR=/path/to/pcie_ar_configs
export SGLANG_PCIE_AR_SOURCE_PATH=/path/to/symm_allreduce_ext.cu
export SGLANG_PCIE_AR_BUILD_DIR=/path/to/build
```

SGLang server 示例：

```bash
CUDA_VISIBLE_DEVICES=4,5 \
SGLANG_USE_PCIE_CUSTOM_ALL_REDUCE=1 \
SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0 \
SGLANG_PCIE_AR_ENABLE_PDL=1 \
python -m sglang.launch_server \
  --model-path /path/to/Qwen3.5-35B-A3B-FP8 \
  --tp 2 \
  --cuda-graph-bs 1 2 4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64
```

## Smoke test

预编译后先跑轻量 correctness：

```bash
cd /path/to/sglang
CUDA_VISIBLE_DEVICES=4,5 \
PYTHONPATH=python python scripts/pcie_allreduce/smoke_pcie_allreduce.py \
  --tp 2 \
  --hidden 2048 \
  --dtype bf16 \
  --batches 1,4,16,64
```

通过标准：

- 所有 rank `correct_all=True`。
- 没有 hang。
- latency 数值和 standalone microbench 同量级。

## GSM8K 精度验证

用于确认替换 allreduce 后不破坏生成正确性：

```bash
cd /path/to/sglang
python scripts/pcie_allreduce/run_sglang_pcie_ar_gsm8k.py \
  --gpus 4,5 \
  --model-path /path/to/Qwen3.5-35B-A3B-FP8 \
  --max-new-tokens 10000
```

建议至少跑三组：

```text
baseline
ipc-pdl-off
ipc-pdl-on
```

如果 harness 环境检查不适配当前机器，可在脚本中使用已有环境并跳过无关检查，但不要跳过结果一致性检查。

## E2E benchmark 和 Nsight timeline

标准 cc16/cc14 矩阵：

```bash
cd /path/to/sglang
python scripts/pcie_allreduce/run_sglang_pcie_ar_e2e.py \
  --phase all \
  --gpus 4,5 \
  --model-path /path/to/Qwen3.5-35B-A3B-FP8
```

默认比较：

```text
baseline: SGLang default behavior
ipc-pdl-off: SGLANG_USE_PCIE_CUSTOM_ALL_REDUCE=1, PDL off
ipc-pdl-on:  SGLANG_USE_PCIE_CUSTOM_ALL_REDUCE=1, PDL on
```

默认产物：

```text
plan.json
preflight.json
results.jsonl
cases/<case>/server/server.log
cases/<case>/benchmark/benchmark.json
cases/<case>/benchmark/requests.jsonl
cases/<case>/nsys/<case>-decode-5steps.nsys-rep
```

只跑计划或检查环境：

```bash
python scripts/pcie_allreduce/run_sglang_pcie_ar_e2e.py --phase plan
python scripts/pcie_allreduce/run_sglang_pcie_ar_e2e.py --phase preflight
```

只跑一个 benchmark：

```bash
python scripts/pcie_allreduce/run_sglang_pcie_ar_e2e.py \
  --phase benchmark \
  --only cc16-cg16-bs16-baseline
```

只抓一个 Nsight：

```bash
python scripts/pcie_allreduce/run_sglang_pcie_ar_e2e.py \
  --phase nsys \
  --only cc16-cg16-bs16-ipc-pdl-on
```

## Pareto 曲线

用于比较 SGLang baseline 和 PCIe AR 在不同并发下的 TPOT/throughput：

```bash
cd /path/to/sglang
python scripts/pcie_allreduce/run_sglang_pcie_ar_pareto.py \
  --gpus 4,5 \
  --model-path /path/to/Qwen3.5-35B-A3B-FP8 \
  --concurrency-grid 1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,96,128
```

建议同时跑两种 CUDA graph 配置：

- exact graph：每个测试 concurrency 都在 cuda graph batch size 里。
- pow2 graph：只配置 1/2/4/8/16/32/64/128，模拟更粗的默认配置。

输出中重点看：

- `TPOT p50/p90`
- `1000 / TPOT_p50_ms`
- output-only throughput per GPU
- input+output throughput per GPU

## 性能分析口径

Nsight 里要区分两个概念：

- `GPU time sum`：多个 rank、多个 stream 上 kernel duration 的累加，适合看 kernel 成本。
- `step wall time`：成对 rank 的 decode graph 起止时间，才是端到端 TPOT 的直接解释变量。

TP2 下两个 rank 通常不对称。判断收益时优先看慢 rank 的 custom allreduce 总和，因为它更接近 critical path。

示例 cc16 分析中：

```text
selected step wall time:      8106.40 us -> 7810.00 us, delta -296.40 us
TP custom AR sum both ranks:  1295.81 us -> 697.44 us, delta -598.37 us
TP custom AR slow-rank sum:   845.60 us  -> 549.92 us, delta -295.68 us
```

这里通信 kernel 成本接近 1.9x，但 decode step 只提升约 3.8%。正确解释是：慢 rank 通信关键路径缩短约 296 us，和 step wall time 缩短一致；不能把双 rank GPU time sum 当作端到端收益。

## PDL 注意事项

PDL 正确用法：

- consumer 可以提前启动并做 prologue。
- consumer 真正读取 producer 输出前必须调用 `cudaGridDependencySynchronize()`。
- producer 的 `cudaTriggerProgrammaticLaunchCompletion()` release 点必须保证 output 已完整 ready。
- release 最好放在 allreduce result 写完后，而不是等 scratch reset、epoch update、ack 等收尾全部完成后。

当前 SGLang 集成里 PDL 是 runtime 参数：

```bash
export SGLANG_PCIE_AR_ENABLE_PDL=1
```

是否有可见 overlap，需要用 Nsight timeline 验证。若 SGLang overlap schedule 被禁用，即使 PDL 开关打开，也可能看不到 consumer kernel 提前启动。

## 常见问题

### 找不到 policy

日志会出现类似：

```text
PCIe custom allreduce disabled: no policy rows found
```

处理：

- 确认 `SGLANG_PCIE_AR_CONFIG_DIR`。
- 确认 `policy.json` 的 `tp/hidden/batch/dtype` 覆盖当前 shape。
- 确认 dtype 是 `bf16/fp16/fp32` 之一。

### 编译失败

处理：

- 先跑 `scripts/pcie_allreduce/prebuild_pcie_allreduce.py --verbose`。
- 确认 CUDA toolkit、PyTorch、GPU 架构匹配。
- 如果源码在外部 repo，确认 `SGLANG_PCIE_AR_SOURCE_PATH` 指向存在的 `.cu` 文件。

### 首次请求很慢

通常是首次 torch extension 编译。先执行预编译，或者复用固定的 `SGLANG_PCIE_AR_BUILD_DIR`。

### 端到端没走 PCIe AR

处理：

- 确认 `SGLANG_USE_PCIE_CUSTOM_ALL_REDUCE=1`。
- 设置 `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0` 让路径更明确。
- 检查 server log 是否有 `[AR] Using PCIe custom allreduce` 和初始化日志。
- 检查 shape 是否在 policy 中；miss 会 fallback。

### PDL 没看到收益

这不一定是 kernel 错误。需要确认：

- consumer 是否真的有可 overlap 的 prologue。
- SGLang overlap schedule 是否启用。
- Nsight 中 consumer kernel 是否在 allreduce release 后、producer cleanup 前启动。

## 提交流程建议

干净分支只提交源码、policy seed、脚本和文档，不提交 benchmark 输出：

```bash
git status --short
git add \
  python/sglang/srt/distributed/device_communicators/custom_all_reduce.py \
  python/sglang/srt/environ.py \
  python/sglang/srt/distributed/device_communicators/pcie_custom_all_reduce.py \
  python/sglang/srt/distributed/device_communicators/pcie_ar_config.py \
  python/sglang/srt/distributed/device_communicators/pcie_ar/ \
  python/sglang/srt/distributed/device_communicators/pcie_ar_configs/ \
  scripts/pcie_allreduce/ \
  docs/advanced_features/pcie_custom_allreduce_integration.md
git commit -m "Add PCIe custom allreduce backend"
```

推送到个人 fork：

```bash
git remote add samuellees git@github.com:samuellees/sglang.git
git push -u samuellees pcie-custom-allreduce-integration-20260712
```
