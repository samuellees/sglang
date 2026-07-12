# SGLang PCIe Custom AllReduce

This directory contains the SGLang integration for the PCIe low-latency
custom allreduce backend.

For the full integration guide, including how to sync the standalone
allreduce repository into SGLang, generate policy files, run correctness
checks, and capture Nsight timelines, see:

```text
docs/advanced_features/pcie_custom_allreduce_integration.md
```

## Generate A Seed Policy

```bash
PYTHONPATH=python python scripts/pcie_allreduce/tune_pcie_allreduce.py \
  --tp 2 \
  --hidden 2048 \
  --dtype bf16 \
  --batch-grid default,fine \
  --max-bytes 8388608
```

The script defaults to the combined grid:

```text
1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,128,256,512
```

The named grids can also be used separately:

```bash
--batch-grid default
--batch-grid fine
```

where `default` means:

```text
1,2,4,8,16,32,64,128,256,512
```

and `fine` means:

```text
1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64
```

The policy is written under:

```text
python/sglang/srt/distributed/device_communicators/pcie_ar_configs/
```

Override the config root with:

```bash
export SGLANG_PCIE_AR_CONFIG_DIR=/path/to/pcie_ar_configs
```

## Use In SGLang

Enable the backend with:

```bash
export SGLANG_USE_PCIE_CUSTOM_ALL_REDUCE=1
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0
```

The second line is not strictly required when PCIe AR is enabled, but it makes
the intended path explicit when comparing with SGLang's default JIT backend.

Optional switches:

```bash
export SGLANG_PCIE_AR_ENABLE_PDL=1
export SGLANG_PCIE_AR_MAX_SIZE_BYTES=8388608
export SGLANG_PCIE_AR_VERBOSE_BUILD=1
```

First launch builds the extension from:

```text
python/sglang/srt/distributed/device_communicators/pcie_ar/symm_allreduce_ext.cu
```

To move compile cost out of the first SGLang warmup, prebuild the extension:

```bash
PYTHONPATH=python python scripts/pcie_allreduce/prebuild_pcie_allreduce.py
```

For a small correctness smoke after prebuild:

```bash
CUDA_VISIBLE_DEVICES=4,5 \
PYTHONPATH=python python scripts/pcie_allreduce/smoke_pcie_allreduce.py \
  --tp 2 \
  --hidden 2048 \
  --dtype bf16 \
  --batches 1,4,20,64
```

Default build cache:

```text
${SGLANG_CACHE_DIR:-~/.cache/sglang}/pcie_allreduce/build
```

Use `SGLANG_PCIE_AR_BUILD_DIR` to override it.

## Qwen3.5-35B TP2 SGLang E2E + NSYS Matrix

The end-to-end runner compares three backend modes:

```text
baseline: SGLang default behavior
ipc-pdl-off: SGLANG_USE_PCIE_CUSTOM_ALL_REDUCE=1, PDL off
ipc-pdl-on:  SGLANG_USE_PCIE_CUSTOM_ALL_REDUCE=1, PDL on
```

It runs these CUDA graph / IFB cases:

```text
cc16, cuda graph batch size 16
cc14, cuda graph batch size 16 only
cc14, cuda graph batch sizes 14 and 16
```

Each case emits one benchmark and one 5-step decode Nsight Systems report, so
the default matrix is 9 benchmark runs plus 9 nsys runs. The benchmark client is
dependency-free and sends 160 streaming completion requests with target
input/output lengths of 1024/1024.

```bash
python scripts/pcie_allreduce/run_sglang_pcie_ar_e2e.py \
  --phase all \
  --gpus 4,5 \
  --model-path /path/to/Qwen3.5-35B-A3B-FP8
```

Useful shorter commands:

```bash
# Write the 9-case plan only.
python scripts/pcie_allreduce/run_sglang_pcie_ar_e2e.py --phase plan

# Check paths, nsys binary and visible CUDA devices.
python scripts/pcie_allreduce/run_sglang_pcie_ar_e2e.py --phase preflight

# Run only the cc16 baseline benchmark.
python scripts/pcie_allreduce/run_sglang_pcie_ar_e2e.py \
  --phase benchmark \
  --only cc16-cg16-bs16-baseline

# Capture only the IPC+PDL cc14 exact-graph nsys timeline.
python scripts/pcie_allreduce/run_sglang_pcie_ar_e2e.py \
  --phase nsys \
  --only cc14-cg14-16-bs14-exact-ipc-pdl-on
```

Default output root is under the current working tree unless `--output-root` is
specified:

```text
task-output/sglang-pcie-ar-qwen35-35b-tp2-ifb-<date>/
```

Important files:

```text
plan.json
preflight.json
results.jsonl
cases/<case>/server/server.log
cases/<case>/benchmark/benchmark.json
cases/<case>/benchmark/requests.jsonl
cases/<case>/profile/profile_client.json
cases/<case>/nsys/<case>-decode-5steps.nsys-rep
```
