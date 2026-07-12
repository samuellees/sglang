#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _default_build_dir() -> Path:
    configured = os.environ.get("SGLANG_PCIE_AR_BUILD_DIR")
    if configured:
        return Path(configured)
    cache_root = os.environ.get("SGLANG_CACHE_DIR", str(Path.home() / ".cache" / "sglang"))
    return Path(cache_root) / "pcie_allreduce" / "build"


def _load_ext(path: Path):
    spec = importlib.util.spec_from_file_location("sglang_pcie_allreduce_ext", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def _parse_batches(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.replace(" ", "").split(",") if item)


def _worker(rank: int, args: argparse.Namespace, port: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=args.tp,
    )

    dtype = _dtype(args.dtype)
    batches = _parse_batches(args.batches)
    ext = _load_ext(args.extension)
    obj = ext.IpcPushAllreduce(
        rank,
        args.tp,
        args.hidden * max(batches),
        torch.empty((), dtype=dtype).element_size(),
        args.max_blocks,
    )
    handles = [None for _ in range(args.tp)]
    dist.all_gather_object(handles, obj.share_storage())
    obj.post_init(handles)
    dist.barrier()

    expected_sum = args.tp * (args.tp + 1) // 2
    for batch in batches:
        inp = torch.full(
            (batch, args.hidden),
            rank + 1,
            device="cuda",
            dtype=dtype,
        )
        out = torch.empty_like(inp)
        obj.all_reduce_v2(
            inp,
            out,
            inp.numel(),
            args.blocks,
            args.threads,
            batch >= args.stream_batch_threshold,
            args.pdl,
            args.pdl,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out,
            torch.full_like(out, expected_sum),
            rtol=0,
            atol=0,
        )

    obj.close()
    dist.destroy_process_group()
    if rank == 0:
        print(
            f"PCIe allreduce smoke passed: tp={args.tp} hidden={args.hidden} "
            f"dtype={args.dtype} batches={','.join(map(str, batches))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small correctness smoke for the PCIe allreduce extension."
    )
    parser.add_argument("--tp", type=int, default=2, choices=(2, 4, 8))
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--batches", default="1,4,20,64")
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--max-blocks", type=int, default=128)
    parser.add_argument("--stream-batch-threshold", type=int, default=4)
    parser.add_argument("--pdl", action="store_true")
    parser.add_argument(
        "--gpus",
        default="",
        help="Optional CUDA_VISIBLE_DEVICES value, e.g. 4,5.",
    )
    parser.add_argument(
        "--extension",
        type=Path,
        default=_default_build_dir() / "sglang_pcie_allreduce_ext.so",
    )
    args = parser.parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if not args.extension.exists():
        raise SystemExit(
            f"extension not found: {args.extension}\n"
            "Run scripts/pcie_allreduce/prebuild_pcie_allreduce.py first."
        )
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    mp.spawn(_worker, args=(args, _free_port()), nprocs=args.tp)


if __name__ == "__main__":
    main()
