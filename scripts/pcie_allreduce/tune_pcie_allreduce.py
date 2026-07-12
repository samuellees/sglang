#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - allows config generation in minimal envs
    torch = None


def _load_config_module():
    repo_root = Path(__file__).resolve().parents[2]
    path = (
        repo_root
        / "python"
        / "sglang"
        / "srt"
        / "distributed"
        / "device_communicators"
        / "pcie_ar_config.py"
    )
    spec = importlib.util.spec_from_file_location("pcie_ar_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pcie_ar_config = _load_config_module()


def _device_name() -> str:
    if torch is not None and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "unknown_cuda_device"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a seed policy for SGLang PCIe custom allreduce."
    )
    parser.add_argument("--tp", type=int, required=True, choices=(2, 4, 8))
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument(
        "--batch-grid",
        default="default,fine",
        help="default, fine, default,fine, or a comma-separated batch list.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=8 * 1024 * 1024,
        help="Only emit rows whose input tensor size is <= this limit.",
    )
    parser.add_argument("--max-blocks", type=int, default=128)
    parser.add_argument(
        "--config-dir", type=Path, default=pcie_ar_config.default_config_dir()
    )
    parser.add_argument("--device-name", default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print generated policy JSON after writing it.",
    )
    args = parser.parse_args()

    device_name = args.device_name or _device_name()
    batch_grid = pcie_ar_config.parse_batch_grid(args.batch_grid)
    rows = pcie_ar_config.generate_heuristic_rows(
        args.tp,
        args.hidden,
        args.dtype,
        batch_grid,
        args.max_blocks,
        args.max_bytes,
    )
    if not rows:
        raise SystemExit("no rows generated; check --hidden, --dtype and --max-bytes")

    policy_path = pcie_ar_config.policy_path_for_shape(
        args.config_dir,
        device_name,
        args.tp,
        args.hidden,
        args.dtype,
        profile=args.profile,
    )
    profile_name = args.profile or (
        f"{pcie_ar_config.sanitize_device_name(device_name)}-tp{args.tp}-h{args.hidden}-{args.dtype}"
    )
    pcie_ar_config.write_policy(policy_path, profile_name, rows)

    metadata = {
        "schema_version": 1,
        "profile_name": profile_name,
        "device_name": device_name,
        "tp": args.tp,
        "hidden": args.hidden,
        "dtype": args.dtype,
        "batch_grid": [row.batch for row in rows],
        "max_bytes": args.max_bytes,
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "note": "heuristic seed; replace rows with measured winners after benchmark tuning",
    }
    metadata_path = policy_path.parent / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(policy_path)
    if args.print_json:
        print(policy_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
