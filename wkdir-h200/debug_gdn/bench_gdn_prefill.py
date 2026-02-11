"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import numpy as np
import torch
from einops import rearrange

from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.testing.utils import bench_gpu_time
from sglang.srt.layers.attention.fla.chunk import ChunkGatedDeltaRuleFunction
from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd


def gdn_flops(
    total_seq_len: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    num_seqs: int,
) -> int:
    """
    Calculate FLOPs for Gated Delta Rule (GDN) attention.

    Delta Rule formula:
        state_t = alpha_t * state_{t-1} + beta_t * (k_t @ v_t^T)
        output_t = q_t @ state_t

    Matrix multiplications per token per head:
    1. k @ v^T (outer product): 2 * d^2 FLOPs
    2. q @ state: 2 * d^2 FLOPs

    Note: alpha/beta gating are element-wise scalar multiplications,
    not counted in TFLOPS.
    """
    num_o_heads = max(num_q_heads, num_v_heads)

    # k @ v^T (outer product): 2 * d^2 per token per head
    outer_product_flops = 2 * total_seq_len * num_o_heads * head_size * head_size

    # q @ state: 2 * d^2 per token per head
    output_flops = 2 * total_seq_len * num_o_heads * head_size * head_size

    total_flops = outer_product_flops + output_flops
    return total_flops


def gdn_bytes(
    total_seq_len: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    num_seqs: int,
    dtype: torch.dtype,
) -> int:
    """
    Calculate memory bytes for GDN attention.

    Includes:
    - Q, K, V tensors (input)
    - Output tensor
    - State tensor (float32)
    - Alpha, Beta tensors (optional, float32)
    """
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads
    elem_size = dtype.itemsize

    # Input tensors
    q_bytes = total_seq_len * num_q_heads * head_size * elem_size
    k_bytes = total_seq_len * num_k_heads * head_size * elem_size
    v_bytes = total_seq_len * num_v_heads * head_size * elem_size

    # Output tensor
    o_bytes = total_seq_len * num_o_heads * head_size * elem_size

    # State tensor (float32)
    state_bytes = num_seqs * num_sab_heads * head_size * head_size * 4

    # Alpha and Beta (float32)
    alpha_bytes = total_seq_len * num_sab_heads * 4
    beta_bytes = total_seq_len * num_sab_heads * 4

    total_bytes = (
        q_bytes + k_bytes + v_bytes + o_bytes + state_bytes + alpha_bytes + beta_bytes
    )
    return total_bytes


def bench_gdn_triton(
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    use_alpha: bool = True,
    use_beta: bool = True,
):
    """Benchmark Triton GDN kernel."""
    total_seq_len = batch_size * seq_len
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    # Triton kernel requires batch_size=1 when using cu_seqlens
    # So we reshape: [B, L, ...] -> [1, B*L, ...]
    q = torch.randn(1, total_seq_len, num_q_heads, head_size, dtype=dtype, device="cuda")
    k = torch.randn(1, total_seq_len, num_k_heads, head_size, dtype=dtype, device="cuda")
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
    v = torch.randn(1, total_seq_len, num_v_heads, head_size, dtype=dtype, device="cuda")

    cu_seqlens = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int64, device="cuda"
    )

    # g should be in log space for Triton (will be exp'd internally)
    g = (
        torch.randn(1, total_seq_len, num_sab_heads, dtype=torch.float32, device="cuda") * 0.1 - 3.0
        if use_alpha
        else None
    )
    beta = (
        torch.randn(1, total_seq_len, num_sab_heads, dtype=torch.float32, device="cuda") * 0.5
        if use_beta
        else None
    )

    # Set initial state to zero to eliminate input influence
    # Note: initial_state shape is [num_sequences_in_batch, ...], where num_sequences = batch_size
    initial_state = torch.zeros(
        batch_size, num_sab_heads, head_size, head_size, dtype=torch.float32, device="cuda"
    )
    initial_state_indices = torch.arange(batch_size, dtype=torch.int64, device="cuda")
    scale = 1.0
    use_qk_l2norm_in_kernel = True

    # Warmup
    ChunkGatedDeltaRuleFunction.apply(
        q, k, v, g, beta, scale, initial_state, initial_state_indices, cu_seqlens, use_qk_l2norm_in_kernel
    )
    torch.cuda.synchronize()

    # Benchmark
    times = bench_gpu_time(
        lambda: ChunkGatedDeltaRuleFunction.apply(
            q, k, v, g, beta, scale, initial_state, initial_state_indices, cu_seqlens, use_qk_l2norm_in_kernel
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        enable_cupti=True,
    )

    median_ms = np.median(times)

    # Calculate metrics
    flops = gdn_flops(
        total_seq_len, num_q_heads, num_k_heads, num_v_heads, head_size, batch_size
    )
    bytes_accessed = gdn_bytes(
        total_seq_len,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        batch_size,
        dtype,
    )

    tflops = flops / median_ms / 1e9
    tb_per_sec = bytes_accessed / median_ms / 1e9

    return {
        "median_ms": median_ms,
        "tflops": tflops,
        "tb_per_sec": tb_per_sec,
    }


def bench_gdn_flashinfer(
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    use_alpha: bool = True,
    use_beta: bool = True,
):
    """Benchmark FlashInfer GDN kernel."""
    total_seq_len = batch_size * seq_len
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    # Create inputs
    q = torch.randn(total_seq_len, num_q_heads, head_size, dtype=dtype, device="cuda")
    k = torch.randn(total_seq_len, num_k_heads, head_size, dtype=dtype, device="cuda")
    k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
    v = torch.randn(total_seq_len, num_v_heads, head_size, dtype=dtype, device="cuda")

    cu_seqlens = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int64, device="cuda"
    )

    # FlashInfer expects alpha in exp space (already exp'd)
    alpha = (
        torch.exp(torch.randn(total_seq_len, num_sab_heads, dtype=torch.float32, device="cuda") * 0.1 - 3.0)
        if use_alpha
        else None
    )
    beta = (
        torch.randn(total_seq_len, num_sab_heads, dtype=torch.float32, device="cuda") * 0.5
        if use_beta
        else None
    )

    # Set initial state to zero to eliminate input influence
    initial_state = torch.zeros(
        batch_size, num_sab_heads, head_size, head_size, dtype=torch.float32, device="cuda"
    )

    # Pre-allocate outputs
    output = torch.empty(
        total_seq_len, num_o_heads, head_size, dtype=dtype, device="cuda"
    )
    output_state = torch.empty(
        batch_size,
        num_sab_heads,
        head_size,
        head_size,
        dtype=torch.float32,
        device="cuda",
    )

    # Warmup
    chunk_gated_delta_rule(
        q, k, v, alpha, beta, None, initial_state, True, cu_seqlens, False, output, output_state
    )
    torch.cuda.synchronize()

    # Benchmark
    times = bench_gpu_time(
        lambda: chunk_gated_delta_rule(
            q,
            k,
            v,
            alpha,
            beta,
            None,
            initial_state,
            True,
            cu_seqlens,
            False,
            output,
            output_state,
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        enable_cupti=True,
    )

    median_ms = np.median(times)

    # Calculate metrics
    flops = gdn_flops(
        total_seq_len, num_q_heads, num_k_heads, num_v_heads, head_size, batch_size
    )
    bytes_accessed = gdn_bytes(
        total_seq_len,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        batch_size,
        dtype,
    )

    tflops = flops / median_ms / 1e9
    tb_per_sec = bytes_accessed / median_ms / 1e9

    return {
        "median_ms": median_ms,
        "tflops": tflops,
        "tb_per_sec": tb_per_sec,
    }


def compare_accuracy(
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
):
    """Compare accuracy between Triton and FlashInfer."""
    total_seq_len = batch_size * seq_len
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    # Create shared inputs
    q_fi = torch.randn(total_seq_len, num_q_heads, head_size, dtype=dtype, device="cuda")
    k_fi = torch.randn(total_seq_len, num_k_heads, head_size, dtype=dtype, device="cuda")
    k_fi = torch.nn.functional.normalize(k_fi, p=2.0, dim=-1)
    v_fi = torch.randn(total_seq_len, num_v_heads, head_size, dtype=dtype, device="cuda")

    cu_seqlens = torch.arange(
        0, batch_size * seq_len + 1, seq_len, dtype=torch.int64, device="cuda"
    )

    # Generate alpha in log space, then convert for FlashInfer
    alpha_log = torch.randn(total_seq_len, num_sab_heads, dtype=torch.float32, device="cuda") * 0.1 - 3.0
    alpha_fi = torch.exp(alpha_log)  # FlashInfer expects exp'd values
    beta_fi = torch.randn(total_seq_len, num_sab_heads, dtype=torch.float32, device="cuda") * 0.5
    
    # Set initial state to zero to eliminate input influence
    initial_state_fi = torch.zeros(
        batch_size, num_sab_heads, head_size, head_size, dtype=torch.float32, device="cuda"
    )

    # FlashInfer
    output_fi = torch.empty(total_seq_len, num_o_heads, head_size, dtype=dtype, device="cuda")
    output_state_fi = torch.empty(
        batch_size, num_sab_heads, head_size, head_size, dtype=torch.float32, device="cuda"
    )
    chunk_gated_delta_rule(
        q_fi, k_fi, v_fi, alpha_fi, beta_fi, None, initial_state_fi, True, cu_seqlens, False, output_fi, output_state_fi
    )

    # Triton (convert format and use log-space alpha)
    # Triton requires batch_size=1 when using cu_seqlens, so reshape to [1, B*L, ...]
    q_triton = rearrange(q_fi, "(b l) h d -> 1 (b l) h d", b=batch_size)
    k_triton = rearrange(k_fi, "(b l) h d -> 1 (b l) h d", b=batch_size)
    v_triton = rearrange(v_fi, "(b l) h d -> 1 (b l) h d", b=batch_size)
    g_triton = rearrange(alpha_log, "(b l) h -> 1 (b l) h", b=batch_size)  # Use log-space for Triton
    beta_triton = rearrange(beta_fi, "(b l) h -> 1 (b l) h", b=batch_size)

    initial_state_triton = initial_state_fi.clone()
    initial_state_indices = torch.arange(batch_size, dtype=torch.int64, device="cuda")
    
    output_triton, output_state_triton = ChunkGatedDeltaRuleFunction.apply(
        q_triton, k_triton, v_triton, g_triton, beta_triton, 1.0,
        initial_state_triton, initial_state_indices, cu_seqlens, True
    )
    output_triton = rearrange(output_triton, "1 (b l) h d -> (b l) h d", b=batch_size)

    # Debug: print shapes
    print(f"DEBUG: output_state_fi.shape = {output_state_fi.shape}")
    print(f"DEBUG: output_state_triton.shape = {output_state_triton.shape}")
    
    # Compute accuracy metrics for output o
    max_diff_o = torch.max(torch.abs(output_triton - output_fi)).item()
    mean_diff_o = torch.mean(torch.abs(output_triton - output_fi)).item()
    
    pred_flat = output_triton.flatten().float()
    target_flat = output_fi.flatten().float()
    ss_res = torch.sum((target_flat - pred_flat) ** 2)
    ss_tot = torch.sum((target_flat - torch.mean(target_flat)) ** 2)
    r2_o = (1 - ss_res / ss_tot).item()

    # Compute accuracy metrics for output state h
    # output_state_fi: [batch, num_heads, head_size, head_size]
    # output_state_triton: [batch, num_heads, head_size, head_size]
    # Only compare the last batch's state (since we only have 1 batch in this case)
    if batch_size == 1:
        max_diff_h = torch.max(torch.abs(output_state_triton[0:1] - output_state_fi)).item()
        mean_diff_h = torch.mean(torch.abs(output_state_triton[0:1] - output_state_fi)).item()
        
        pred_flat_h = output_state_triton[0:1].flatten().float()
        target_flat_h = output_state_fi.flatten().float()
        ss_res_h = torch.sum((target_flat_h - pred_flat_h) ** 2)
        ss_tot_h = torch.sum((target_flat_h - torch.mean(target_flat_h)) ** 2)
        r2_h = (1 - ss_res_h / ss_tot_h).item()
    else:
        # For multiple batches, we need to properly map them
        max_diff_h = float('nan')
        mean_diff_h = float('nan')
        r2_h = float('nan')

    return {
        "max_diff_o": max_diff_o,
        "mean_diff_o": mean_diff_o,
        "r2_o": r2_o,
        "max_diff_h": max_diff_h,
        "mean_diff_h": mean_diff_h,
        "r2_h": r2_h,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark GDN Prefill Kernel")
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--seq-len", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--num-q-heads", type=int, default=16)
    parser.add_argument("--num-k-heads", type=int, default=16)
    parser.add_argument("--num-v-heads", type=int, default=32)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument(
        "--dtype", type=str, choices=["float16", "bfloat16"], default="bfloat16"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["qwen3-next", "custom"],
        default="custom",
        help="Use preset config. qwen3-next: q=k=16, v=32, d=128",
    )
    args = parser.parse_args()

    # Apply preset configurations
    if args.preset == "qwen3-next":
        # Qwen3-Next-80B-A3B linear attention config (GVA)
        args.num_q_heads = 16
        args.num_k_heads = 16
        args.num_v_heads = 32
        args.head_size = 128

    # Check SM90 support
    device_capability = torch.cuda.get_device_capability()
    if device_capability[0] < 9:
        print(f"Current device capability: {device_capability}")
        print("GDN requires SM90 (Hopper) or later. Exiting...")
        return

    dtype = getattr(torch, args.dtype)

    print(
        f"GDN Prefill Benchmark (heads: q={args.num_q_heads}, k={args.num_k_heads}, v={args.num_v_heads}, d={args.head_size}, dtype={args.dtype})"
    )
    print("=" * 90)
    print(f"{'batch':>6} {'seq_len':>8} {'kernel':>12} | {'time(ms)':>10} {'TFLOPS':>10} {'TB/s':>10} {'speedup':>10}")
    print("=" * 90)

    for batch_size in args.batch_size:
        for seq_len in args.seq_len:
            result_fi = bench_gdn_flashinfer(
                batch_size=batch_size,
                seq_len=seq_len,
                num_q_heads=args.num_q_heads,
                num_k_heads=args.num_k_heads,
                num_v_heads=args.num_v_heads,
                head_size=args.head_size,
                dtype=dtype,
            )
            result_triton = bench_gdn_triton(
                batch_size=batch_size,
                seq_len=seq_len,
                num_q_heads=args.num_q_heads,
                num_k_heads=args.num_k_heads,
                num_v_heads=args.num_v_heads,
                head_size=args.head_size,
                dtype=dtype,
            )
            speedup = result_triton["median_ms"] / result_fi["median_ms"]
            
            print(
                f"{batch_size:>6} {seq_len:>8} {'FlashInfer':>12} | "
                f"{result_fi['median_ms']:>10.3f} {result_fi['tflops']:>10.2f} {result_fi['tb_per_sec']:>10.2f} {'':>10}"
            )
            print(
                f"{'':>6} {'':>8} {'Triton':>12} | "
                f"{result_triton['median_ms']:>10.3f} {result_triton['tflops']:>10.2f} {result_triton['tb_per_sec']:>10.2f} {speedup:>9.2f}x"
            )
            print("-" * 90)

    print("=" * 90)
    
    # Accuracy comparison
    print("\nAccuracy Comparison (Triton vs FlashInfer) - Initial State = 0:")
    print("=" * 110)
    print("Output O Comparison:")
    print("-" * 110)
    print(f"{'batch':>6} {'seq_len':>8} {'max_diff_o':>15} {'mean_diff_o':>15} {'R2_o':>15}")
    print("-" * 110)
    
    for batch_size in args.batch_size:
        for seq_len in args.seq_len:
            acc = compare_accuracy(
                batch_size=batch_size,
                seq_len=seq_len,
                num_q_heads=args.num_q_heads,
                num_k_heads=args.num_k_heads,
                num_v_heads=args.num_v_heads,
                head_size=args.head_size,
                dtype=dtype,
            )
            print(
                f"{batch_size:>6} {seq_len:>8} {acc['max_diff_o']:>15.6e} {acc['mean_diff_o']:>15.6e} {acc['r2_o']:>15.8f}"
            )
    
    print("-" * 110)
    print("\nOutput State H Comparison:")
    print("-" * 110)
    print(f"{'batch':>6} {'seq_len':>8} {'max_diff_h':>15} {'mean_diff_h':>15} {'R2_h':>15}")
    print("-" * 110)
    
    for batch_size in args.batch_size:
        for seq_len in args.seq_len:
            acc = compare_accuracy(
                batch_size=batch_size,
                seq_len=seq_len,
                num_q_heads=args.num_q_heads,
                num_k_heads=args.num_k_heads,
                num_v_heads=args.num_v_heads,
                head_size=args.head_size,
                dtype=dtype,
            )
            print(
                f"{batch_size:>6} {seq_len:>8} {acc['max_diff_h']:>15.6e} {acc['mean_diff_h']:>15.6e} {acc['r2_h']:>15.8f}"
            )
    
    print("-" * 110)


if __name__ == "__main__":
    main()
