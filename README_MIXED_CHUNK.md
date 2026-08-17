# Qwen3.5 mixed-chunk integration and validation

This change consolidates the mixed-chunk implementation and its integration
fixes into one commit on top of public SGLang commit
`593777c0465890fb466d0008a2d0417e48fcea61` (`#33527`). Intermediate source
feature, fix, documentation, and merge commits are intentionally represented
by the single consolidated change.

Existing behavior is preserved unless `--enable-mixed-chunk` is selected.
The final GSM8K scores are 0.9773 for TP16 MTP3+1 and 0.9795 for DP4xTP4 +
EP16 MTP2+1.

## Execution model

SGLang lays out a mixed batch as prompt/extend tokens followed by one resident
decode token per decode request:

```text
[prefill request tokens ........................][resident decode tokens]
                 |                                          |
                 v                                          v
          context/extend kernels                    decode kernels (TP)
```

The optimization routes the two slices independently inside TRTLLM MHA and
FlashInfer GDN, then restores the original token order. It does not overlap
prefill and decode and does not change scheduler token accounting.

The optimized GDN/decode split is enabled only for the validated TP path.
DP-attention/WideEP keeps a single legacy GDN extend launch for mixed batches,
and high-local-GQA MHA keeps one full context-FMHA launch. Those narrow
fallbacks are required for recurrent-state correctness; pure prefill, pure
decode, and mixed-off execution retain their original paths.

## Integration fixes

The source changes did not compose safely without the following fixes:

- use one authoritative `mixed_decode_batch_size` geometry across regular,
  TBO, and CUDA-graph batch objects;
- support Qwen3.5 NextN top-k-1 ReplaySSM mixed admissions;
- resolve deferred speculative sequence lengths before mixed allocation;
- reuse the speculative KV reserve instead of allocating/freeing the resident
  tail twice;
- isolate TRTLLM mixed-decode workspace, page tables, and FMHA counters;
- drain the previous DeepEPv2 result before entering a speculative global
  extend/mixed batch, without disabling steady-state decode overlap;
- keep DP-attention high-local-GQA MHA on its correctness-proven full mixed
  context launch;
- keep DP-attention mixed GDN on its correctness-proven single extend launch.

The last item fixed the most subtle failure. A 512-example run could begin with
correct answers and then corrupt resident recurrent state immediately after a
new admission wave. The same failure occurred with MTP off, MTP2+1, and both
`no_buffer` and `extra_buffer`, which ruled out speculative acceptance and
cache policy. Restricting the split to non-DP attention repaired both 512-item
gates and the complete 1,319-item run.

## Validation dependencies

The final runs used FlashInfer PR 4358 at
`23922f9a336e05839d33b0cc822267773560e0a0` and WideEP DeepEPv2 at
`01dc3aaac82068020353dce2c302e38153c0bfaa`. Model, container, cache, and
scheduler locations are environment-specific and intentionally omitted.

For reproduction, pin the code and dependency revisions, use the topology and
workload parameters below, and run each ratio as a separate matched A/B pair.
The model and runtime artifacts should be supplied through the local execution
environment.

## Accuracy

Both accuracy runs use the complete GSM8K set and finish every example.

| Topology | MTP | Mixed | Client threads | Examples | Score | Stop | Truncated | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TP16 | 3+1 | on | 128 | 1319/1319 | **0.9772555** | 1.0000 | 0 | 0 |
| DP4xTP4 + EP16 | 2+1 | on | 512 | 1319/1319 | **0.9795299** | 1.0000 | 0 | 0 |

The TP result predates the final DP-only runtime fixes. Those changes are gated
on DP attention or DeepEPv2 mixed execution, so the final branch takes the
identical TP code path. The WideEP result uses the final validated code.

Before the full WideEP run, two 512-example repeated-admission gates completed:

| Topology | MTP | Score | Stop | Truncated | Errors |
|---|---:|---:|---:|---:|---:|
| DP4xTP4 + EP16 | off | 0.9824219 | 0.9980469 | 0.0019531 | 0 |
| DP4xTP4 + EP16 | 2+1 | 0.9843750 | 0.9980469 | 0.0019531 | 0 |

## Performance methodology

- GB300, 16 GPUs, FP8 weights, FP8 KV cache.
- Random prompts: nominal ISL 8192, OSL 1024, exact 1024-token output,
  `ignore_eos=true`.
- Ratio 1.0 fixes prompt length at 8192; ratio 0.8 samples the requested range.
- Each high-concurrency result completes `3 x CC` requests with request rate
  `inf`; CC below is both configured and actual maximum concurrency.
- `Total tok/s/GPU = total_token_throughput / 16`.

### TP16 MTP3+1 CC128 gate, ratio 0.8

| Mixed | Actual CC | Completed | Total tok/s | Total tok/s/GPU | Output tok/s | TTFT ms | TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 128 | 384/384 | 22,957.14 | 1,434.82 | 2,569.32 | 4,947.26 | 42.62 |
| on | 128 | 384/384 | 22,126.38 | 1,382.90 | 2,476.35 | 6,310.82 | 43.44 |

Mixed chunk changes total throughput by **-3.62%** at this point.

### TP16 CC512 matrix

| MTP | Ratio | Actual CC | Mixed off total tok/s | Mixed on total tok/s | On tok/s/GPU | On TPOT ms | On vs off |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 1.0 | 512 | 29,447.92 | 25,239.92 | 1,577.50 | 156.03 | **-14.29%** |
| off | 0.8 | 512 | 26,085.82 | 24,614.13 | 1,538.38 | 158.92 | **-5.64%** |
| 2+1 | 1.0 | 512 | 30,785.97 | 25,364.21 | 1,585.26 | 153.35 | **-17.61%** |
| 2+1 | 0.8 | 512 | 28,793.22 | 25,137.63 | 1,571.10 | 154.31 | **-12.70%** |

All rows completed 1536/1536 requests with zero errors.

### DP4xTP4 attention + EP16 WideEP matrix

| MTP | Ratio | Actual CC | Mixed off total tok/s | Mixed on total tok/s | On tok/s/GPU | On TPOT ms | On vs off |
|---|---:|---:|---:|---:|---:|---:|---:|
| off | 1.0 | 768 | 42,468.28 | 38,746.19 | 2,421.64 | 150.47 | **-8.76%** |
| off | 0.8 | 768 | 35,353.32 | 30,796.10 | 1,924.76 | 195.25 | **-12.89%** |
| 2+1 | 1.0 | 640 | 39,109.68 | 29,567.83 | 1,847.99 | 170.18 | **-24.40%** |
| 2+1 | 0.8 | 768 | 37,449.69 | 28,462.76 | 1,778.92 | 216.99 | **-24.00%** |

The MTP2 ratio-1.0 mixed server does not fit at CC768: DeepGEMM requested a
3.94 GiB allocation with only about 0.9 GiB free. The matched 128-aligned
capacity is CC640; both on and off rows above use CC640. Every listed final run
completed all requests with zero errors.

## Result interpretation

Mixed chunk is correct after this integration, but it is not a throughput win
for these exact current-stack points. TP penalties are smallest at ratio 0.8
and CC128. WideEP penalties are larger because the DP recurrent-state and
high-local-GQA correctness fallbacks deliberately retain the legacy mixed
extend/context kernels. Therefore:

- keep mixed chunk off for production WideEP performance on this branch;
- use the TP split only after validating the target workload, not as a global
  default;
- the next WideEP optimization should first construct rank-local mixed
  geometry/state ownership, then re-enable split GDN and split MHA under DP
  attention with the 512-wave and full GSM8K gates above.

These results do not reproduce the older isolated split-branch 12K gain because
the consolidated source stack also carries fused GDN decode, split-K GEMM,
ReplaySSM, newer FlashInfer, and different graph/runtime behavior. The table is
the authoritative matched A/B result for the integrated branch.

## Tests

Focused tests cover mixed geometry propagation, GDN routing policy, TRTLLM MHA
layout, speculative KV ownership, and the DeepEPv2 scheduling boundary:

```text
python/sglang/srt/layers/attention/test_gdn_prefill_backend_policy.py
test/registered/unit/layers/test_trtllm_mha_mixed_batch.py
test/registered/unit/managers/test_scheduler_decision_batch_params.py
```

`py_compile` and `git diff --check` pass locally. GPU-backed behavior was
validated by the distributed runs summarized above.
