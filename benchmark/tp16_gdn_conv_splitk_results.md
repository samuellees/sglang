# TP16 GDN Conv1D fusion and split-K GEMM results

> Historical measurements: for the current Qwen3.8 strict single-variable,
> two-run results, see [`qwen3_8_optimizations.md`](qwen3_8_optimizations.md).

This note records the previously measured benefits of two independent decode
optimizations integrated on this branch. The numbers are intentionally kept
separate: the split-K GEMM result is a controlled A/B, while the available GDN
Conv1D fusion comparison is cross-build evidence and is not a strict
single-variable attribution.

## Test setup

- Hardware: NVIDIA GB300, four trays / 16 GPUs in one NVL72 topology domain.
- Parallelism: TP16, PP1, EP1.
- Model traffic: exact random ISL 8192 / OSL 1024.
- Weight and attention KV formats: FP8 weights and FP8 E4M3 KV cache.
- Linear-attention decode: FlashInfer GDN under full CUDA Graph.
- Throughput below is aggregate output tokens/second for the TP16 server.

The companion `benchmark/pareto_bcg_decode_full_cc1_128.yaml` explicitly
enables both optimizations.

## GDN projection-unpack + Conv1D fusion

The fused Triton path combines QKVZ/BA unpacking and the indexed causal Conv1D
state update. It removes one kernel launch and avoids writing the unfiltered
QKV intermediate to HBM and reading it back for every GDN layer and decode
step. It also produces aligned B/A tensors required by the FlashInfer GDN
backend. Unsupported tensor contracts retain an explicit correctness fallback.

Enable it with:

```bash
export SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV=1
```

The following points compare the earlier staged TP16 stack with the later
integrated stack containing GDN fusion. Both used FP8 KV, exact 8192/1024
traffic, node-local staged weights, and NCCL 2.30.7.

| Concurrent requests | Earlier stack output tok/s | Integrated stack output tok/s | Observed change |
| ---: | ---: | ---: | ---: |
| 256 | 1,587.861 | 1,623.029 | +2.215% |
| 480 | 1,705.320 | 1,720.436 | +0.886% |

The two-point geometric-mean change is **+1.548%**.

This is directional rather than strict A/B evidence. The earlier run used the
v0.5.16 image, while the integrated run used a newer nightly image and also
contained the split-K integration. The steady CC256/480 decode shapes are
outside the split-K low-M allowlist, which reduces that confound, but runtime
transitions and the remaining build differences prevent assigning the full
change solely to GDN fusion. A strict fusion-only result requires the same
commit, image, nodes, and config with only
`SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV=0/1` changed.

Correctness coverage includes the TP16-local GDN shape, indexed state updates,
BF16/FP16 cases, CUDA Graph replay, aligned B/A repair, and explicit fallback.
Production logs confirmed that the fused backend was selected across the GDN
layers during TP16 graph capture and serving.

## FlashInfer split-K BF16 GEMM

The split-K implementation is opt-in:

```bash
python -m sglang.launch_server \
  ... \
  --bf16-gemm-backend flashinfer_pr4266
```

It uses PDL-enabled, measured tactics only for the exact TP16 Qwen3.5 low-M
allowlist: `M={1,2,4,8,16,24,32}`, `N={256,512,2304,2560}`, and `K=8192`.
All other shapes use the original SGLang TGV/cuBLAS path.

The end-to-end result below is an order-balanced controlled comparison. Both
sides used the same four trays, image, SGLang commit, staged model, NCCL 2.30.7
runtime, serving configuration, and GDN fusion. Only the BF16 GEMM backend
changed. The paired estimate is `sqrt((O1/B1) * (O2/B2))`.

| CC | Baseline B1 tok/s | Optimized O1 tok/s | Optimized O2 tok/s | Baseline B2 tok/s | Paired throughput change | Paired mean-ITL change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 103.068 | 103.016 | 103.402 | 103.024 | +0.157% | -0.199% |
| 2 | 197.995 | 198.266 | 199.158 | 197.723 | +0.431% | -0.445% |
| 4 | 352.820 | 367.050 | 366.935 | 352.732 | +4.030% | -4.446% |
| 8 | 551.665 | 568.390 | 566.554 | 552.535 | +2.784% | -3.284% |
| 16 | 749.900 | 769.156 | 768.610 | 752.224 | +2.373% | -2.910% |
| 24 | 879.428 | 905.411 | 905.082 | 880.180 | +2.892% | -3.607% |
| 32 | 988.635 | 1,003.030 | 1,003.592 | 987.661 | +1.535% | -1.996% |

- Geometric mean over CC1-32: **+2.021% output throughput**.
- Geometric mean over the intended beneficial range CC4-32:
  **+2.719% output throughput**.
- CC1-2 are effectively neutral; every repeated CC4-32 point improved.
- The isolated selected GEMMs improved by 1.261x-1.632x, but the smaller
  end-to-end gain is expected because GEMM is only part of the decode graph.

### Fallback regression and accuracy gates

- CC64 deliberately falls outside the allowlist. Baseline was 1,197.843
  output tok/s and optimized was 1,196.075 output tok/s (**-0.148%**), with
  mean ITL changing by +0.195%. This is noise-level parity and verifies the
  original-path fallback.
- GSM8K: **0.9809741248**, 1,289/1,314 correct, with max running requests and
  decode graph capped at 32.
- GPQA: **0.875 mean accuracy**, 1,386/1,584 stochastic samples across eight
  repeats (`temperature=0.6`, `top_p=0.95`).

The accuracy runs are functional gates for the optimized stack; they are not
claimed as accuracy improvements over the baseline.
