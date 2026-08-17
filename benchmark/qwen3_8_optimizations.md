# Qwen3.8 TP16 decode optimizations

This note summarizes two independent end-to-end optimizations for Qwen3.8:

1. allowlisted low-M BF16 Split-K GEMM; and
2. fused GDN projection unpack plus indexed Conv1D state update.

The two optimizations were measured separately. In each experiment, the other
optimization was explicitly disabled, so the gains below must not be added
together without a separate combined-stack measurement.

## Test protocol

- Hardware: 16 NVIDIA GB300 GPUs in one NVL72 topology domain.
- Parallelism: TP16, PP1, EP1.
- Workload: exact random ISL 8192 / OSL 1024.
- Concurrency: 1, 2, 4, 8, 16, 32, and 64.
- Precision: FP8 weights and FP8 E4M3 attention KV cache.
- CUDA Graph: breakable prefill graph and full decode graph.
- Cache: radix cache disabled; Mamba strategy `no_buffer`.
- Speculative decoding: disabled.
- Sampling: two warmup waves and five measured requests per concurrency unit.
- Metric: end-to-end total tokens/second/GPU, including input and output
  tokens.

The optimized arm was run twice with the same source revision and serving
configuration. Each round completed 635/635 measured requests with zero
request errors and no prefill or decode graph fallback. The baseline is a
common controlled run with both experimental optimizations disabled.

## Low-M BF16 Split-K GEMM

On Blackwell, `--bf16-gemm-backend auto` selects the optimized CuTe DSL
backend. The Split-K path is enabled by default on this branch and can be
controlled explicitly with:

```bash
export SGLANG_ENABLE_BF16_SPLITK_GEMM=1  # enable
export SGLANG_ENABLE_BF16_SPLITK_GEMM=0  # disable for an A/B baseline
```

The tuned allowlist is limited to `M={1,2,4,8,16,24,32}`,
`N={256,512,2304,2560}`, and `K=8192`. Other shapes retain the existing
backend.

The Conv1D fusion was disabled in both Split-K runs.

| CC | Baseline | Split-K run 1 | Run 1 vs baseline | Split-K run 2 | Run 2 vs baseline | Run 2 vs run 1 | Two-run mean vs baseline |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 68.155 | 71.052 | +4.250% | 71.395 | +4.753% | +0.482% | +4.502% |
| 2 | 124.530 | 126.302 | +1.423% | 126.248 | +1.380% | -0.043% | +1.401% |
| 4 | 224.445 | 234.180 | +4.337% | 233.039 | +3.829% | -0.487% | +4.083% |
| 8 | 358.950 | 368.493 | +2.659% | 368.558 | +2.677% | +0.018% | +2.668% |
| 16 | 506.319 | 523.025 | +3.300% | 522.417 | +3.180% | -0.116% | +3.240% |
| 32 | 692.064 | 711.714 | +2.839% | 713.923 | +3.158% | +0.310% | +2.999% |
| 64 | 873.588 | 872.557 | -0.118% | 873.741 | +0.018% | +0.136% | -0.050% |

Values are total tokens/second/GPU. Across CC1-32, where the tuned low-M
allowlist applies, the arithmetic mean gain is **+3.134% in run 1**, **+3.163%
in run 2**, and **+3.149% for the two-run mean**. CC64 is neutral because its
main GEMMs fall outside the allowlist.

Run-to-run throughput differences stay within **-0.487% to +0.482%**, with a
mean absolute difference of 0.227%. The improvement at CC1-32 is therefore
larger than the observed optimized-run variation.

## Fused GDN projection unpack and Conv1D

The fused path combines QKVZ/BA unpacking with the indexed causal Conv1D state
update. It removes an intermediate QKV write/read and a separate kernel
launch, while producing the aligned B/A tensors required by the FlashInfer GDN
backend.

It is enabled by default for supported CUDA decode shapes and can be controlled
explicitly with:

```bash
export SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV=1  # enable
export SGLANG_ENABLE_GDN_DECODE_FUSED_PROJ_CONV=0  # disable for an A/B baseline
```

The Split-K GEMM path was disabled in both Conv1D-fusion runs.

| CC | Baseline | Fusion run 1 | Run 1 vs baseline | Fusion run 2 | Run 2 vs baseline | Run 2 vs run 1 | Two-run mean vs baseline |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 68.155 | 70.487 | +3.421% | 70.568 | +3.540% | +0.115% | +3.481% |
| 2 | 124.530 | 125.220 | +0.554% | 125.985 | +1.169% | +0.611% | +0.861% |
| 4 | 224.445 | 229.645 | +2.317% | 229.732 | +2.356% | +0.038% | +2.336% |
| 8 | 358.950 | 364.172 | +1.455% | 363.839 | +1.362% | -0.092% | +1.408% |
| 16 | 506.319 | 511.138 | +0.952% | 509.339 | +0.597% | -0.352% | +0.774% |
| 32 | 692.064 | 697.762 | +0.823% | 698.089 | +0.871% | +0.047% | +0.847% |
| 64 | 873.588 | 878.042 | +0.510% | 877.739 | +0.475% | -0.034% | +0.493% |

Values are total tokens/second/GPU. The unweighted mean gain across CC1-64 is
**+1.457% for the two-run mean**. The benefit is largest at CC1 and decays as
batch-scaled work dominates. Run-to-run throughput differences remain within
**-0.352% to +0.611%**, with a signed mean of +0.048%.

The fused path was positively identified in both runs, its fallback marker was
absent, and all observed decode batches used the captured full graph.

### MTP limitation

The current fusion gate requires strict `DECODE` mode. Qwen MTP steady state
uses `TARGET_VERIFY` and draft modes, so enabling the environment variable does
not dispatch this fusion in the MTP main loop. No MTP performance gain should
be attributed to this optimization until those forward modes are supported and
measured separately.

## Practical recommendation

- Keep both optimizations enabled for non-MTP Qwen3.8 serving on Blackwell.
- Expect roughly 3.1% average gain from Split-K over CC1-32 and neutral behavior
  once the workload falls outside its allowlist.
- Expect roughly 0.5-3.5% from GDN Conv1D fusion across CC1-64, averaging 1.46%
  in this sweep.
- Use the environment variables above as independent kill switches when
  debugging correctness or performance regressions.
- Do not assume the isolated gains are additive; benchmark the combined stack
  before reporting a combined benefit.
