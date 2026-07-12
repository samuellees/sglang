#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_source() -> Path:
    return (
        _repo_root()
        / "python"
        / "sglang"
        / "srt"
        / "distributed"
        / "device_communicators"
        / "pcie_ar"
        / "symm_allreduce_ext.cu"
    )


def _default_build_dir() -> Path:
    configured = os.environ.get("SGLANG_PCIE_AR_BUILD_DIR")
    if configured:
        return Path(configured)
    cache_root = os.environ.get("SGLANG_CACHE_DIR", str(Path.home() / ".cache" / "sglang"))
    return Path(cache_root) / "pcie_allreduce" / "build"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prebuild the SGLang PCIe custom allreduce torch extension."
    )
    parser.add_argument("--source", type=Path, default=_default_source())
    parser.add_argument("--build-dir", type=Path, default=_default_build_dir())
    parser.add_argument("--name", default="sglang_pcie_allreduce_ext")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"source not found: {args.source}")

    args.build_dir.mkdir(parents=True, exist_ok=True)
    module = load(
        name=args.name,
        sources=[str(args.source)],
        build_directory=str(args.build_dir),
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=args.verbose,
    )
    print(f"built {args.name} in {args.build_dir}")
    print(getattr(module, "__file__", "module loaded"))


if __name__ == "__main__":
    main()
