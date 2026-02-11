from __future__ import annotations

import argparse
import math
import os
import random
import time

import torch
import pytest

import functools

from sglang.srt.layers.attention.fla.chunk import ChunkGatedDeltaRuleFunction
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as chunk_gated_delta_rule_fla

def compare(sgl, fi, fla, name=""):
    if name:
        print(f"\n{name}:")
    print(f"  SGLang  shape={sgl.shape}, range=({sgl.min():.4f}, {sgl.max():.4f}), mean={sgl.mean():.4f}")
    print(f"  FlashInfer shape={fi.shape}, range=({fi.min():.4f}, {fi.max():.4f}), mean={fi.mean():.4f}")
    # print(f"  FLA shape={fla.shape}, range=({fla.min():.4f}, {fla.max():.4f}), mean={fla.mean():.4f}")
    
    # Calculate differences
    diff = torch.abs(sgl - fi)
    rel_diff = diff / (torch.abs(sgl) + 1e-8)
    print(f"  Abs Error: mean={diff.mean():.6f}, max={diff.max():.6f}; Rel Error: mean={rel_diff.mean():.6f}, max={rel_diff.max():.6f}")

def _test_prefill_kernel(
    qkv_factory,
    dtype: str,
    num_q_heads_origin: int,
    num_k_heads_origin: int,
    num_v_heads_origin: int,
    head_size: int,
    seq_lens: list[int],
    scale: float,
    tp_size: int = 1,
    seed: int | None = None,
    compare_output: bool = False,
    compare_states: bool = False,
):
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_q_heads = num_q_heads_origin // tp_size
    num_k_heads = num_k_heads_origin // tp_size
    num_v_heads = num_v_heads_origin // tp_size
    num_seqs = len(seq_lens)
    total_seqlen = sum(seq_lens)
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = max(num_q_heads, num_v_heads)

    dtype = getattr(torch, dtype)
    kv_dtype = torch.float32
    device = torch.device("cuda")
    with device:
        q, k, v = qkv_factory(
            seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, dtype
        )
        # l2 norm k to avoid numerical instability
        k = torch.nn.functional.normalize(k, p=2.0, dim=-1)
        cu_seq_lens = torch.tensor([0, seq_lens[0]], dtype=torch.int64)
        alpha = torch.rand(total_seqlen, num_sab_heads) 
        beta = torch.rand(total_seqlen, num_sab_heads)
        # Set alpha to be close to 1.0 to better match the distribution of real gating values, 
        # A smaller alpha will significantly decrease the weight of previous tokens' states
        alpha = alpha  * 0.004 + 0.996
        alpha = torch.log(alpha)

    our_o = torch.empty(
        [total_seqlen, num_o_heads, head_size], dtype=q.dtype, device=q.device
    )
    our_state = torch.empty(
        (num_seqs, num_sab_heads, head_size, head_size),
        dtype=torch.float32,
        device=q.device,
    )


    # Triton SGLang
    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    v = v.unsqueeze(0)
    alpha = alpha.unsqueeze(0)
    beta = beta.unsqueeze(0)
    initial_states = torch.zeros(
        (num_seqs, num_sab_heads, head_size, head_size),
        dtype=dtype,
        device=q.device,
    )
    initial_states_indices = torch.arange(num_seqs, dtype=torch.int64, device=device)
    
    # Warmup for ChunkGatedDeltaRuleFunction
    for _ in range(3):
        ref_o, ref_state = ChunkGatedDeltaRuleFunction.apply(
            q,
            k,
            v,
            alpha,
            beta,
            scale,
            initial_states,
            initial_states_indices,
            cu_seq_lens,
            False,
        )
    torch.cuda.synchronize()
    
    # Benchmark ChunkGatedDeltaRuleFunction
    num_iters = 100
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iters):
        ref_o, ref_state = ChunkGatedDeltaRuleFunction.apply(
            q,
            k,
            v,
            alpha,
            beta,
            scale,
            initial_states,
            initial_states_indices,
            cu_seq_lens,
            False,
        )
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / num_iters
    
    ref_o = ref_o.squeeze(0)
    ref_state = ref_state.squeeze(0)
    ref_o = ref_o.to(q.dtype)
    ref_state = ref_state.to(kv_dtype)


    # Warmup for chunk_gated_delta_rule_fla
    o_fla, h_fla = chunk_gated_delta_rule_fla(
        q.bfloat16(),
        k.bfloat16(),
        v.bfloat16(),
        alpha,
        beta,
        scale,
        None,
        output_final_state=True,
        cu_seqlens=cu_seq_lens.to(torch.int32),
        use_qk_l2norm_in_kernel=False,
    )
    torch.cuda.synchronize()
    
    o_fla = o_fla.squeeze(0)
    h_fla = h_fla.squeeze(0)
    

    # FlashInfer
    our_o.fill_(float("nan"))
    our_state.fill_(float("nan"))
    q = q.squeeze(0)
    k = k.squeeze(0)
    v = v.squeeze(0)
    beta = beta.squeeze(0)
    for _ in range(3):
        alpha_fi = torch.exp(alpha.squeeze(0))
        chunk_gated_delta_rule(
            q,
            k,
            v,
            alpha_fi,
            beta,
            scale,
            None,
            True,
            cu_seq_lens,
            False,
            output=our_o,
            output_state=our_state,
        )

    torch.cuda.synchronize()
    num_iters = 100
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iters):
        alpha_fi = torch.exp(alpha.squeeze(0))
        chunk_gated_delta_rule(
            q,
            k,
            v,
            alpha_fi,
            beta,
            scale,
            None,
            True,
            cu_seq_lens,
            False,
            output=our_o,
            output_state=our_state,
        )
    torch.cuda.synchronize()
    flashinfer_time = (time.time() - start_time) / num_iters

    # postprocessing raw output, ref_state is v-major, our_state is k-major, unify to v-major for testing
    our_state = our_state.transpose(-1, -2)

    # Compare results if requested
    if compare_output:
        compare(ref_o, our_o, o_fla, name="Output Comparison")
    
    if compare_states:
        compare(ref_state, our_state, h_fla, name="State Comparison")

    # Return benchmark results
    return {
        'seqlen': total_seqlen,
        'num_qk_heads': num_q_heads_origin,
        'num_v_heads': num_v_heads_origin,
        'head_size': head_size,
        'tp_size': tp_size,
        'triton_time': triton_time * 1000,  # ms
        'flashinfer_time': flashinfer_time * 1000,  # ms
        'speedup': triton_time / flashinfer_time
    }


def multidist_randu(num_dists, dim, mean_mean=0.0, mean_std=1.0, lower=-1.0, upper=1.0):
    means = torch.distributions.Normal(mean_mean, mean_std).sample((num_dists,))
    data = torch.distributions.Uniform(means + lower, means + upper).sample((dim,))
    return data.T.contiguous()

def qkv_factory(
    seq_lens, num_q_heads, num_k_heads, num_v_heads, head_size, dtype=torch.float16
):
    # qkv_rng = functools.partial(multidist_randn, mean_std=0.1)
    qkv_rng = functools.partial(multidist_randu, mean_std=0.05, lower=-0.25, upper=0.25)

    total_seq_lens = sum(seq_lens)
    q = qkv_rng(total_seq_lens * num_q_heads, head_size)
    k = qkv_rng(total_seq_lens * num_k_heads, head_size)
    v = qkv_rng(total_seq_lens * num_v_heads, head_size)

    q = q.reshape(total_seq_lens, num_q_heads, head_size).to(dtype).contiguous()
    k = k.reshape(total_seq_lens, num_k_heads, head_size).to(dtype).contiguous()
    v = v.reshape(total_seq_lens, num_v_heads, head_size).to(dtype).contiguous()

    return q, k, v

def test_prefill_kernel_nonfull(
    seed: int = int(os.environ.get("SEED", "0")),
    compare_output: bool = False,
    compare_states: bool = False,
):
    scale = "auto"
    head_size = 128
    # num_q_heads, num_k_heads, num_v_heads = 16, 16, 32
    dtype = "bfloat16"
    scale = 1.0 / math.sqrt(head_size) if scale == "auto" else scale
    
    results = []
    for num_q_heads, num_k_heads, num_v_heads in [(16, 16, 32), (16, 16, 64)]:
        for tp_size in [1, 2, 4]:
            for seqlen in [1024, 2048, 4096, 8192, 16384, 32768, 65536]:
                seq_lens = [seqlen]
                print(f"\nTesting seqlen={seqlen}, tp_size={tp_size}, num_q_heads={num_q_heads}, num_k_heads={num_k_heads}, num_v_heads={num_v_heads} ... ")
                result = _test_prefill_kernel(
                    qkv_factory,
                    dtype,
                    num_q_heads,
                    num_k_heads,
                    num_v_heads,
                    head_size,
                    seq_lens,
                    scale,
                    tp_size,
                    seed,
                    compare_output,
                    compare_states,
                )
                results.append(result)
    
    # Print table header
    print("\n" + "="*115)
    print(f"{'SeqLen':<10} {'TP':<5} {'QK Heads':<10} {'V Heads':<10} {'HeadSize':<10} {'Triton(ms)':<15} {'FlashInfer(ms)':<15} {'Speedup':<10}")
    print("="*115)
    
    # Print table rows
    for r in results:
        print(f"{r['seqlen']:<10} {r['tp_size']:<5} {r['num_qk_heads']:<10} {r['num_v_heads']:<10} {r['head_size']:<10} "
              f"{r['triton_time']:<15.4f} {r['flashinfer_time']:<15.4f} {r['speedup']:<10.2f}")
    
    print("="*115)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark GDN prefill kernels')
    parser.add_argument('--check-output', action='store_true', help='Compare output tensors')
    parser.add_argument('--check-states', action='store_true', help='Compare state tensors')
    parser.add_argument('--seed', type=int, default=int(os.environ.get("SEED", "0")), help='Random seed')
    args = parser.parse_args()
    
    test_prefill_kernel_nonfull(
        seed=args.seed,
        compare_output=args.check_output,
        compare_states=args.check_states,
    )