#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import html
import json
import math
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import run_sglang_pcie_ar_e2e as e2e


DEFAULT_OUT = (
    e2e.REPO_ROOT.parent.parent
    / "task-output"
    / "sglang-pcie-ar-qwen35-35b-tp2-pareto-20260711"
)

CONCURRENCY_GRID = (
    1,
    2,
    4,
    8,
    12,
    16,
    20,
    24,
    28,
    32,
    36,
    40,
    44,
    48,
    52,
    56,
    60,
    64,
    96,
    128,
)

BACKENDS = {
    "baseline": e2e.Backend("baseline", use_pcie_ar=False, pdl=False),
    "ours_pdl": e2e.Backend("ipc_pdl", use_pcie_ar=True, pdl=True),
}

GRAPH_MODES = {
    "pow2": (1, 2, 4, 8, 16, 32, 64, 128),
    "exact": CONCURRENCY_GRID,
}

SERIES = (
    ("exact", "baseline", "SGLang baseline / exact graph", "#5b6472", ""),
    ("pow2", "baseline", "SGLang baseline / pow2 graph", "#5b6472", "7 5"),
    ("exact", "ours_pdl", "Ours IPC+PDL / exact graph", "#0f8b8d", ""),
    ("pow2", "ours_pdl", "Ours IPC+PDL / pow2 graph", "#0f8b8d", "7 5"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen3.5-35B TP2 SGLang Pareto benchmarks for SGLang "
            "baseline and the PCIe IPC+PDL allreduce backend."
        )
    )
    parser.add_argument("--python", type=Path, default=e2e.DEFAULT_PYTHON)
    parser.add_argument("--sglang-root", type=Path, default=e2e.REPO_ROOT)
    parser.add_argument("--model-path", default=e2e.DEFAULT_MODEL)
    parser.add_argument("--served-model-name", default="qwen35-35b")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=33300)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--phase",
        choices=("all", "benchmark", "plot", "plan", "preflight"),
        default="all",
    )
    parser.add_argument(
        "--graph-mode",
        choices=("all", *GRAPH_MODES.keys()),
        default="all",
    )
    parser.add_argument(
        "--backend",
        choices=("all", *BACKENDS.keys()),
        default="all",
    )
    parser.add_argument(
        "--grid",
        default=",".join(str(v) for v in CONCURRENCY_GRID),
        help="Comma-separated CUDA graph/concurrency grid.",
    )
    parser.add_argument(
        "--client-grid",
        default="",
        help=(
            "Optional comma-separated client concurrency grid. When set, --grid "
            "still controls the CUDA graph list and server max-running-requests."
        ),
    )
    parser.add_argument(
        "--prompt-multiplier",
        type=int,
        default=10,
        help="num_prompts = concurrency * prompt_multiplier.",
    )
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--ready-timeout-seconds", type=int, default=1800)
    parser.add_argument("--bench-timeout-seconds", type=int, default=900)
    parser.add_argument("--server-exit-timeout-seconds", type=int, default=120)
    parser.add_argument("--max-tokens-context", type=int, default=8192)
    parser.add_argument("--mem-fraction-static", default="")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--allow-no-gpu", action="store_true")
    parser.add_argument("--extra-server-arg", action="append", default=[])
    return parser.parse_args()


def parse_grid(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("--grid must contain at least one concurrency")
    if any(value <= 0 for value in values):
        raise ValueError("--grid values must be positive")
    return tuple(dict.fromkeys(values))


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def free_port(start: int) -> int:
    for port in range(start, start + 400):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port found from {start}")


def selected_graph_modes(args: argparse.Namespace) -> list[str]:
    if args.graph_mode == "all":
        return list(GRAPH_MODES)
    return [args.graph_mode]


def selected_backends(args: argparse.Namespace) -> list[str]:
    if args.backend == "all":
        return list(BACKENDS)
    return [args.backend]


def graph_bs_for_mode(mode: str, grid: tuple[int, ...]) -> tuple[int, ...]:
    if mode == "pow2":
        max_cc = max(grid)
        values = [value for value in GRAPH_MODES[mode] if value <= max_cc]
        if values[-1] < max_cc:
            values.append(1 << (max_cc - 1).bit_length())
        return tuple(values)
    if mode == "exact":
        return grid
    raise ValueError(f"unknown graph mode: {mode}")


def group_case(mode: str, backend_name: str, grid: tuple[int, ...]) -> e2e.Case:
    backend = BACKENDS[backend_name]
    graph = e2e.GraphCase(mode, max(grid), graph_bs_for_mode(mode, grid))
    return e2e.Case(backend, graph)


def group_name(mode: str, backend_name: str) -> str:
    return f"qwen35-35b-fp8-tp2-ifb-{mode}-{backend_name}"


def case_name(mode: str, backend_name: str, concurrency: int) -> str:
    return (
        "qwen35-35b-fp8-tp2-ifb"
        f"-{mode}"
        f"-cc{concurrency:03d}"
        f"-{'baseline' if backend_name == 'baseline' else 'ours-ipc-pdl'}"
    )


def rel(path: Path) -> str:
    return str(path.resolve())


def write_plan(
    args: argparse.Namespace,
    graph_grid: tuple[int, ...],
    client_grid: tuple[int, ...],
) -> None:
    plan: list[dict[str, Any]] = []
    for mode in selected_graph_modes(args):
        graph_bs = graph_bs_for_mode(mode, graph_grid)
        for backend_name in selected_backends(args):
            plan.append(
                {
                    "group": group_name(mode, backend_name),
                    "graph_mode": mode,
                    "backend": backend_name,
                    "cuda_graph_bs": list(graph_bs),
                    "max_running_requests": max(graph_grid),
                    "client_concurrency": list(client_grid),
                    "num_prompts_rule": f"concurrency * {args.prompt_multiplier}",
                    "group_dir": rel(args.out_root / "groups" / group_name(mode, backend_name)),
                }
            )
    write_json(
        args.out_root / "plan.json",
        {
            "created_at": now(),
            "model_path": args.model_path,
            "tp": args.tp,
            "gpus": args.gpus,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "groups": plan,
        },
    )
    print(f"plan: {args.out_root / 'plan.json'}", flush=True)


def preflight(args: argparse.Namespace) -> None:
    checks = {
        "created_at": now(),
        "python": str(args.python),
        "python_exists": args.python.exists(),
        "sglang_root": str(args.sglang_root),
        "model_path": args.model_path,
        "model_path_exists": Path(args.model_path).exists(),
        "dev_nvidiactl_exists": Path("/dev/nvidiactl").exists(),
    }
    try:
        probe = (
            "import json, torch; "
            "print(json.dumps({'torch': torch.__version__, "
            "'cuda_available': torch.cuda.is_available(), "
            "'device_count': torch.cuda.device_count()}))"
        )
        output = subprocess.check_output(
            [str(args.python), "-c", probe],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        checks["torch_probe"] = output.strip().splitlines()[-1]
    except Exception as exc:
        checks["torch_probe_error"] = f"{type(exc).__name__}: {exc}"
    write_json(args.out_root / "preflight.json", checks)
    if not args.allow_no_gpu and not checks["dev_nvidiactl_exists"]:
        raise RuntimeError("GPU device files are not visible.")


def case_result_path(args: argparse.Namespace, mode: str, backend_name: str, cc: int) -> Path:
    return args.out_root / "cases" / case_name(mode, backend_name, cc) / "benchmark" / "benchmark.json"


def run_group(
    args: argparse.Namespace,
    *,
    mode: str,
    backend_name: str,
    graph_grid: tuple[int, ...],
    client_grid: tuple[int, ...],
    group_index: int,
) -> None:
    group = group_case(mode, backend_name, graph_grid)
    name = group_name(mode, backend_name)
    group_dir = args.out_root / "groups" / name
    port = free_port(args.base_port + group_index * 20)
    base_url = f"http://{args.host}:{port}"
    proc, cmd, env = e2e.launch_server(args, group, group_dir, profile=False, port=port)
    write_json(
        group_dir / "manifest.json",
        {
            "created_at": now(),
            "group": name,
            "graph_mode": mode,
            "backend": backend_name,
            "base_url": base_url,
            "server_command": cmd,
            "env_subset": e2e.env_subset(env),
            "client_concurrency": list(client_grid),
            "cuda_graph_grid": list(graph_grid),
            "prompt_multiplier": args.prompt_multiplier,
        },
    )
    try:
        print(f"[server] waiting group={name} url={base_url}", flush=True)
        e2e.wait_ready(base_url, args.ready_timeout_seconds, proc)
        for cc in client_grid:
            out_dir = args.out_root / "cases" / case_name(mode, backend_name, cc) / "benchmark"
            done = out_dir / "benchmark.json"
            if args.skip_existing and done.exists():
                print(f"[benchmark] skip {mode} {backend_name} cc={cc}", flush=True)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "created_at": now(),
                "graph_mode": mode,
                "backend": backend_name,
                "concurrency": cc,
                "num_prompts": cc * args.prompt_multiplier,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "base_url": base_url,
                "group": name,
                "server_group_dir": rel(group_dir),
            }
            write_json(out_dir / "manifest.json", manifest)
            print(
                f"[benchmark] {mode} {backend_name} cc={cc} "
                f"prompts={cc * args.prompt_multiplier}",
                flush=True,
            )
            status = "failed"
            error = ""
            try:
                summary = asyncio.run(
                    e2e.run_client(
                        base_url,
                        model=args.served_model_name,
                        concurrency=cc,
                        num_prompts=cc * args.prompt_multiplier,
                        input_len=args.input_len,
                        output_len=args.output_len,
                        timeout_s=args.bench_timeout_seconds,
                        out_dir=out_dir,
                    )
                )
                status = "passed" if summary["failed_requests"] == 0 else "failed"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                (out_dir / "error.txt").write_text(error + "\n")
            append_jsonl(
                args.out_root / "results.jsonl",
                {
                    "created_at": now(),
                    "phase": "benchmark",
                    "graph_mode": mode,
                    "backend": backend_name,
                    "concurrency": cc,
                    "status": status,
                    "error": error,
                    "case_dir": rel(out_dir.parent),
                },
            )
            if proc.poll() is not None:
                raise RuntimeError(f"server {name} exited early with code {proc.returncode}")
    finally:
        rc = e2e.terminate_process(proc, args.server_exit_timeout_seconds)
        append_jsonl(
            args.out_root / "results.jsonl",
            {
                "created_at": now(),
                "phase": "server",
                "group": name,
                "status": "terminated",
                "server_returncode": rc,
                "group_dir": rel(group_dir),
            },
        )


def load_metrics(args: argparse.Namespace, grid: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in GRAPH_MODES:
        for backend_name in BACKENDS:
            for cc in grid:
                path = case_result_path(args, mode, backend_name, cc)
                if not path.exists():
                    continue
                try:
                    payload = json.loads(path.read_text())
                    status = "passed" if payload.get("failed_requests") == 0 else "failed"
                    error = ""
                except Exception as exc:
                    payload = {}
                    status = "failed"
                    error = f"{type(exc).__name__}: {exc}"
                duration_s = payload.get("duration_s")
                total_stream_chunks = payload.get("total_stream_chunks")
                stream_chunks_per_s = payload.get("stream_chunks_per_s")
                tpot_ms_p50 = payload.get("tpot_ms_p50")
                num_prompts = payload.get("num_prompts")
                input_len = payload.get("input_len_target")
                inverse_tpot = None
                if isinstance(tpot_ms_p50, (int, float)) and tpot_ms_p50 > 0:
                    inverse_tpot = 1000.0 / tpot_ms_p50
                output_per_gpu = None
                if isinstance(stream_chunks_per_s, (int, float)):
                    output_per_gpu = stream_chunks_per_s / args.tp
                total_with_input_per_gpu = None
                if (
                    isinstance(duration_s, (int, float))
                    and duration_s > 0
                    and isinstance(total_stream_chunks, (int, float))
                    and isinstance(num_prompts, (int, float))
                    and isinstance(input_len, (int, float))
                ):
                    total_with_input_per_gpu = (
                        total_stream_chunks + num_prompts * input_len
                    ) / duration_s / args.tp
                rows.append(
                    {
                        "graph_mode": mode,
                        "backend": backend_name,
                        "backend_label": (
                            "baseline" if backend_name == "baseline" else "ours_ipc_pdl"
                        ),
                        "concurrency": cc,
                        "cuda_graph_bs": " ".join(str(v) for v in graph_bs_for_mode(mode, grid)),
                        "num_prompts": cc * args.prompt_multiplier,
                        "status": status,
                        "error": error,
                        "case_dir": rel(path.parent.parent),
                        "inverse_tpot_tokens_per_s": inverse_tpot,
                        "output_chunks_per_s_per_gpu": output_per_gpu,
                        "with_input_tokens_per_s_per_gpu": total_with_input_per_gpu,
                        **payload,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ratio(new: float | None, old: float | None) -> float | None:
    if new is None or old in (None, 0):
        return None
    return new / old


def reduction_pct(new: float | None, old: float | None) -> float | None:
    value = ratio(new, old)
    if value is None:
        return None
    return (1.0 - value) * 100.0


def adjacent_rows(rows: list[dict[str, Any]], grid: tuple[int, ...]) -> list[dict[str, Any]]:
    by_key = {
        (row["graph_mode"], row["backend"], int(row["concurrency"])): row
        for row in rows
    }
    output: list[dict[str, Any]] = []
    for mode in GRAPH_MODES:
        for cc in grid:
            base = by_key.get((mode, "baseline", cc), {})
            ours = by_key.get((mode, "ours_pdl", cc), {})
            base_tpot = base.get("tpot_ms_p50")
            ours_tpot = ours.get("tpot_ms_p50")
            base_thr = base.get("stream_chunks_per_s")
            ours_thr = ours.get("stream_chunks_per_s")
            output.append(
                {
                    "graph_mode": mode,
                    "concurrency": cc,
                    "cuda_graph_bs": " ".join(str(v) for v in graph_bs_for_mode(mode, grid)),
                    "baseline_status": base.get("status"),
                    "ours_status": ours.get("status"),
                    "baseline_stream_chunks_per_s": base_thr,
                    "ours_stream_chunks_per_s": ours_thr,
                    "throughput_speedup_ours_vs_baseline": ratio(ours_thr, base_thr),
                    "baseline_tpot_ms_p50": base_tpot,
                    "ours_tpot_ms_p50": ours_tpot,
                    "tpot_reduction_pct_ours_vs_baseline": reduction_pct(ours_tpot, base_tpot),
                    "baseline_inverse_tpot_tokens_per_s": base.get(
                        "inverse_tpot_tokens_per_s"
                    ),
                    "ours_inverse_tpot_tokens_per_s": ours.get(
                        "inverse_tpot_tokens_per_s"
                    ),
                    "baseline_tpot_ms_p90": base.get("tpot_ms_p90"),
                    "ours_tpot_ms_p90": ours.get("tpot_ms_p90"),
                    "baseline_output_chunks_per_s_per_gpu": base.get(
                        "output_chunks_per_s_per_gpu"
                    ),
                    "ours_output_chunks_per_s_per_gpu": ours.get(
                        "output_chunks_per_s_per_gpu"
                    ),
                    "baseline_with_input_tokens_per_s_per_gpu": base.get(
                        "with_input_tokens_per_s_per_gpu"
                    ),
                    "ours_with_input_tokens_per_s_per_gpu": ours.get(
                        "with_input_tokens_per_s_per_gpu"
                    ),
                    "baseline_ttft_ms_p50": base.get("ttft_ms_p50"),
                    "ours_ttft_ms_p50": ours.get("ttft_ms_p50"),
                    "ttft_reduction_pct_ours_vs_baseline": reduction_pct(
                        ours.get("ttft_ms_p50"), base.get("ttft_ms_p50")
                    ),
                    "baseline_e2e_ms_p50": base.get("e2e_ms_p50"),
                    "ours_e2e_ms_p50": ours.get("e2e_ms_p50"),
                    "e2e_reduction_pct_ours_vs_baseline": reduction_pct(
                        ours.get("e2e_ms_p50"), base.get("e2e_ms_p50")
                    ),
                    "baseline_case_dir": base.get("case_dir"),
                    "ours_case_dir": ours.get("case_dir"),
                }
            )
    return output


def value(row: dict[str, Any], key: str) -> float | None:
    item = row.get(key)
    if item is None:
        return None
    try:
        return float(item)
    except (TypeError, ValueError):
        return None


def svg_polyline(points: list[tuple[float, float]], color: str,
                 dash: str = "") -> str:
    if not points:
        return ""
    text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{html.escape(text)}" fill="none" '
        f'stroke="{color}" stroke-width="1.1" stroke-linejoin="round" '
        f'stroke-linecap="round"{dash_attr}/>'
    )


def make_svg(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    title: str,
    output: Path,
    x_guides: tuple[float, ...] = (),
) -> None:
    width, height = 1240, 720
    left, right, top, bottom = 92, 96, 58, 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    available = [
        row
        for row in rows
        if row.get("status") == "passed"
        and value(row, x_key) is not None
        and value(row, y_key) is not None
    ]
    if not available:
        output.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n")
        return
    xs = [value(row, x_key) for row in available]
    ys = [value(row, y_key) for row in available]
    assert all(item is not None for item in xs)
    assert all(item is not None for item in ys)
    x_min = min(xs)  # type: ignore[arg-type]
    x_max = max(xs)  # type: ignore[arg-type]
    y_min = min(ys)  # type: ignore[arg-type]
    y_max = max(ys)  # type: ignore[arg-type]
    if x_key == "concurrency" or x_min >= 0:
        x_min = 0.0
    else:
        x_min = min(0.0, x_min * 0.95)
    x_pad = (x_max - x_min) * 0.08 if x_max > x_min else 1.0
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 1.0
    if x_min < 0:
        x_min -= x_pad
    x_max += x_pad
    y_min = max(0.0, y_min - y_pad)
    y_max += y_pad

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Inter,Arial,sans-serif;fill:#1f2937} .tick{fill:#667085;font-size:12px}",
        ".label{fill:#344054;font-size:14px}.title{font-size:20px;font-weight:650}",
        ".legend{font-size:13px}.guide-label{fill:#475467;font-size:11px}.grid{stroke:#d0d5dd;stroke-width:0.75}.guide{stroke:#98a2b3;stroke-width:1;stroke-dasharray:5 5}.axis{stroke:#475467;stroke-width:1}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{left}" y="32">{html.escape(title)}</text>',
    ]
    for i in range(6):
        x = left + plot_w * i / 5
        raw = x_min + (x_max - x_min) * i / 5
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>')
        parts.append(
            f'<text class="tick" text-anchor="middle" x="{x:.1f}" y="{top + plot_h + 24}">{raw:.0f}</text>'
        )
    for i in range(6):
        y = top + plot_h * i / 5
        raw = y_max - (y_max - y_min) * i / 5
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        parts.append(
            f'<text class="tick" text-anchor="end" x="{left - 10}" y="{y + 4:.1f}">{raw:.2f}</text>'
        )
    for index, guide in enumerate(x_guides):
        if guide < x_min or guide > x_max:
            continue
        x = sx(guide)
        label_y = top + 16 + (index % 2) * 15
        parts.append(
            f'<line class="guide" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>'
        )
        parts.append(
            f'<text class="guide-label" text-anchor="middle" x="{x:.1f}" y="{label_y}">{guide:g} TPS/user</text>'
        )
    parts.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    parts.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    parts.append(f'<text class="label" text-anchor="middle" x="{left + plot_w / 2:.1f}" y="{height - 28}">{html.escape(x_label)}</text>')
    parts.append(
        f'<text class="label" text-anchor="middle" transform="translate(24 {top + plot_h / 2:.1f}) rotate(-90)">{html.escape(y_label)}</text>'
    )
    legend_x = left + plot_w - 330
    legend_y = top + 8
    for index, (mode, backend, label, color, dash) in enumerate(SERIES):
        series_rows = sorted(
            [
                row
                for row in available
                if row["graph_mode"] == mode and row["backend"] == backend
            ],
            key=lambda row: int(row["concurrency"]),
        )
        points = [(sx(value(row, x_key)), sy(value(row, y_key))) for row in series_rows]  # type: ignore[arg-type]
        parts.append(svg_polyline(points, color, dash))
        for row, (x, y) in zip(series_rows, points):
            cc = int(row["concurrency"])
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.1" fill="#ffffff" '
                f'stroke="{color}" stroke-width="1.1"><title>cc={cc} {label}</title></circle>'
            )
            if cc in (1, 16, 32, 64, 128):
                anchor = "end" if x > left + plot_w - 42 else "start"
                dx = -5 if anchor == "end" else 5
                parts.append(
                    f'<text class="tick" text-anchor="{anchor}" x="{x + dx:.1f}" y="{y - 5:.1f}">{cc}</text>'
                )
        ly = legend_y + index * 22
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{legend_x}" y1="{ly}" x2="{legend_x + 26}" y2="{ly}" stroke="{color}" stroke-width="1.1"{dash_attr}/>')
        parts.append(f'<text class="legend" x="{legend_x + 34}" y="{ly + 4}">{html.escape(label)}</text>')
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_report(args: argparse.Namespace, adjacent: list[dict[str, Any]]) -> None:
    report = args.out_root / "pareto-report.html"
    columns = [
        "graph_mode",
        "concurrency",
        "baseline_tpot_ms_p50",
        "ours_tpot_ms_p50",
        "tpot_reduction_pct_ours_vs_baseline",
        "baseline_inverse_tpot_tokens_per_s",
        "ours_inverse_tpot_tokens_per_s",
        "baseline_output_chunks_per_s_per_gpu",
        "ours_output_chunks_per_s_per_gpu",
        "baseline_with_input_tokens_per_s_per_gpu",
        "ours_with_input_tokens_per_s_per_gpu",
        "baseline_stream_chunks_per_s",
        "ours_stream_chunks_per_s",
        "throughput_speedup_ours_vs_baseline",
        "baseline_ttft_ms_p50",
        "ours_ttft_ms_p50",
        "ttft_reduction_pct_ours_vs_baseline",
        "baseline_e2e_ms_p50",
        "ours_e2e_ms_p50",
        "e2e_reduction_pct_ours_vs_baseline",
        "baseline_status",
        "ours_status",
    ]
    headings = {
        "graph_mode": "Graph",
        "concurrency": "CC",
        "baseline_tpot_ms_p50": "Baseline TPOT p50 ms",
        "ours_tpot_ms_p50": "Ours TPOT p50 ms",
        "tpot_reduction_pct_ours_vs_baseline": "TPOT reduction %",
        "baseline_inverse_tpot_tokens_per_s": "Baseline 1000/TPOT",
        "ours_inverse_tpot_tokens_per_s": "Ours 1000/TPOT",
        "baseline_output_chunks_per_s_per_gpu": "Baseline output/GPU",
        "ours_output_chunks_per_s_per_gpu": "Ours output/GPU",
        "baseline_with_input_tokens_per_s_per_gpu": "Baseline input+output/GPU",
        "ours_with_input_tokens_per_s_per_gpu": "Ours input+output/GPU",
        "baseline_stream_chunks_per_s": "Baseline chunks/s",
        "ours_stream_chunks_per_s": "Ours chunks/s",
        "throughput_speedup_ours_vs_baseline": "Throughput speedup",
        "baseline_ttft_ms_p50": "Baseline TTFT p50 ms",
        "ours_ttft_ms_p50": "Ours TTFT p50 ms",
        "ttft_reduction_pct_ours_vs_baseline": "TTFT reduction %",
        "baseline_e2e_ms_p50": "Baseline E2E p50 ms",
        "ours_e2e_ms_p50": "Ours E2E p50 ms",
        "e2e_reduction_pct_ours_vs_baseline": "E2E reduction %",
        "baseline_status": "Baseline status",
        "ours_status": "Ours status",
    }

    def format_cell(item: Any, column: str) -> str:
        if item is None:
            return ""
        if isinstance(item, float):
            if column.endswith("_pct_ours_vs_baseline"):
                return f"{item:.2f}"
            if column.endswith("speedup_ours_vs_baseline"):
                return f"{item:.4f}"
            if "chunks_per_s" in column or "tokens_per_s" in column:
                return f"{item:.1f}"
            return f"{item:.3f}"
        return str(item)

    def numeric_class(item: Any, column: str) -> str:
        if not isinstance(item, float):
            return ""
        if column.endswith("_pct_ours_vs_baseline"):
            return "pos" if item >= 0 else "neg"
        if column.endswith("speedup_ours_vs_baseline"):
            return "pos" if item >= 1.0 else "neg"
        return ""

    def inline_svg(filename: str) -> str:
        path = args.out_root / filename
        if not path.exists():
            return f"<p>Missing plot: {html.escape(str(path))}</p>"
        return path.read_text(encoding="utf-8")

    def avg(column: str, mode: str | None = None, max_cc: int | None = None) -> float:
        values: list[float] = []
        for row in adjacent:
            if mode is not None and row.get("graph_mode") != mode:
                continue
            if max_cc is not None and int(row["concurrency"]) > max_cc:
                continue
            item = row.get(column)
            if isinstance(item, float):
                values.append(item)
        return sum(values) / len(values) if values else 0.0

    rows = []
    for row in adjacent:
        cells = []
        for col in columns:
            item = row.get(col)
            cls = numeric_class(item, col)
            class_attr = f' class="{cls}"' if cls else ""
            cells.append(
                f"<td{class_attr}>{html.escape(format_cell(item, col))}</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    summary_cards = [
        ("Points", str(len(adjacent))),
        ("pow2 TPOT avg reduction", f"{avg('tpot_reduction_pct_ours_vs_baseline', 'pow2'):.2f}%"),
        ("exact TPOT avg reduction", f"{avg('tpot_reduction_pct_ours_vs_baseline', 'exact'):.2f}%"),
        (
            "exact TPOT avg reduction, CC<=64",
            f"{avg('tpot_reduction_pct_ours_vs_baseline', 'exact', 64):.2f}%",
        ),
    ]
    cards_html = "".join(
        f"<div class=\"card\"><div>{html.escape(label)}</div><strong>{html.escape(value)}</strong></div>"
        for label, value in summary_cards
    )
    report.write_text(
        """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Qwen3.5-35B TP2 Pareto Benchmark</title>
<style>
body{font-family:Inter,Arial,sans-serif;margin:28px;color:#1f2937;background:#fff}
h1{font-size:24px;margin:0 0 8px} h2{font-size:18px;margin:30px 0 12px}
p{line-height:1.5;max-width:1120px}.muted{color:#667085;font-size:13px}
.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:20px 0 24px}
.card{border:1px solid #d0d5dd;border-radius:6px;padding:12px;background:#f8fafc}
.card div{font-size:12px;color:#667085}.card strong{display:block;margin-top:6px;font-size:20px}
.plot{margin:14px 0 24px}.plot svg{display:block;max-width:100%;height:auto;border:1px solid #eaecf0}
.table-wrap{overflow:auto;border:1px solid #d0d5dd;border-radius:6px;margin-top:14px;max-height:780px}
table{border-collapse:separate;border-spacing:0;font-size:12px;width:100%}
th,td{border-right:1px solid #d0d5dd;border-bottom:1px solid #d0d5dd;padding:7px 9px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:#fff;z-index:1}
th{position:sticky;top:0;background:#f2f4f7;z-index:2;font-weight:650}
th:first-child{z-index:3;background:#f2f4f7}
.pos{color:#067647}.neg{color:#b42318}.links a{margin-right:14px;color:#175cd3}
</style>
</head>
<body>
<h1>Qwen3.5-35B FP8 TP2 SGLang Pareto Benchmark</h1>
<p>Input/output target: 1024/1024 tokens. Total prompts per point: concurrency * 10. Baseline is SGLang default; ours is IPC custom allreduce with PDL enabled. CUDA graph is evaluated in two modes: power-of-two graph sizes and exact graph sizes.</p>
<div class="cards">"""
        + cards_html
        + """</div>
<p class="links"><a href="pareto-adjacent-comparison.csv">CSV table</a><a href="pareto-adjacent-comparison.xlsx">Excel table</a><a href="pareto-long.csv">Raw long CSV</a></p>
<h2>TPOT vs Throughput</h2>
<div class="plot">"""
        + inline_svg("pareto_tpot_vs_throughput.svg")
        + """</div>
<h2>TPOT vs Concurrency</h2>
<div class="plot">"""
        + inline_svg("tpot_vs_concurrency.svg")
        + """</div>
<h2>Adjacent Comparison Table</h2>
<p class="muted">Positive reduction means ours is faster/lower latency. Throughput speedup above 1.0 means ours has higher output stream throughput.</p>
<div class="table-wrap">
<table>
<thead><tr>"""
        + "".join(f"<th>{html.escape(headings[col])}</th>" for col in columns)
        + "</tr></thead><tbody>"
        + "\n".join(rows)
        + """</tbody></table>
</div>
</body>
</html>
""",
        encoding="utf-8",
    )


def plot_and_summarize(args: argparse.Namespace, grid: tuple[int, ...]) -> None:
    rows = load_metrics(args, grid)
    long_columns = [
        "graph_mode",
        "backend",
        "backend_label",
        "concurrency",
        "cuda_graph_bs",
        "num_prompts",
        "status",
        "duration_s",
        "ok_requests",
        "failed_requests",
        "total_stream_chunks",
        "stream_chunks_per_s",
        "inverse_tpot_tokens_per_s",
        "output_chunks_per_s_per_gpu",
        "with_input_tokens_per_s_per_gpu",
        "ttft_ms_avg",
        "ttft_ms_p50",
        "ttft_ms_p90",
        "tpot_ms_avg",
        "tpot_ms_p50",
        "tpot_ms_p90",
        "e2e_ms_avg",
        "e2e_ms_p50",
        "e2e_ms_p90",
        "case_dir",
        "error",
    ]
    write_csv(args.out_root / "pareto-long.csv", rows, long_columns)
    adjacent = adjacent_rows(rows, grid)
    adjacent_columns = [
        "graph_mode",
        "concurrency",
        "cuda_graph_bs",
        "baseline_status",
        "ours_status",
        "baseline_stream_chunks_per_s",
        "ours_stream_chunks_per_s",
        "throughput_speedup_ours_vs_baseline",
        "baseline_tpot_ms_p50",
        "ours_tpot_ms_p50",
        "tpot_reduction_pct_ours_vs_baseline",
        "baseline_inverse_tpot_tokens_per_s",
        "ours_inverse_tpot_tokens_per_s",
        "baseline_tpot_ms_p90",
        "ours_tpot_ms_p90",
        "baseline_output_chunks_per_s_per_gpu",
        "ours_output_chunks_per_s_per_gpu",
        "baseline_with_input_tokens_per_s_per_gpu",
        "ours_with_input_tokens_per_s_per_gpu",
        "baseline_ttft_ms_p50",
        "ours_ttft_ms_p50",
        "ttft_reduction_pct_ours_vs_baseline",
        "baseline_e2e_ms_p50",
        "ours_e2e_ms_p50",
        "e2e_reduction_pct_ours_vs_baseline",
        "baseline_case_dir",
        "ours_case_dir",
    ]
    write_csv(args.out_root / "pareto-adjacent-comparison.csv", adjacent, adjacent_columns)
    make_svg(
        rows,
        x_key="stream_chunks_per_s",
        y_key="tpot_ms_p50",
        x_label="Output stream chunks per second (higher is better)",
        y_label="TPOT p50 ms (lower is better)",
        title="Qwen3.5-35B FP8 TP2 Pareto: TPOT vs output throughput",
        output=args.out_root / "pareto_tpot_vs_throughput.svg",
    )
    make_svg(
        rows,
        x_key="inverse_tpot_tokens_per_s",
        y_key="output_chunks_per_s_per_gpu",
        x_label="1000 / TPOT p50 (tokens/s, higher is better)",
        y_label="Output throughput per GPU (chunks/s/GPU)",
        title="Qwen3.5-35B FP8 TP2 Pareto: output-only throughput per GPU",
        output=args.out_root / "pareto_inverse_tpot_vs_output_per_gpu.svg",
        x_guides=(50, 60, 100, 150),
    )
    make_svg(
        rows,
        x_key="inverse_tpot_tokens_per_s",
        y_key="with_input_tokens_per_s_per_gpu",
        x_label="1000 / TPOT p50 (tokens/s, higher is better)",
        y_label="Input+output throughput per GPU (tokens/s/GPU)",
        title="Qwen3.5-35B FP8 TP2 Pareto: input+output throughput per GPU",
        output=args.out_root / "pareto_inverse_tpot_vs_with_input_per_gpu.svg",
        x_guides=(50, 60, 100, 150),
    )
    make_svg(
        rows,
        x_key="concurrency",
        y_key="tpot_ms_p50",
        x_label="Client concurrency",
        y_label="TPOT p50 ms (lower is better)",
        title="Qwen3.5-35B FP8 TP2: TPOT vs concurrency",
        output=args.out_root / "tpot_vs_concurrency.svg",
    )
    write_report(args, adjacent)
    print(f"summary: {args.out_root / 'pareto-adjacent-comparison.csv'}", flush=True)
    print(f"plot: {args.out_root / 'pareto_tpot_vs_throughput.svg'}", flush=True)


def main() -> None:
    args = parse_args()
    graph_grid = parse_grid(args.grid)
    client_grid = parse_grid(args.client_grid) if args.client_grid else graph_grid
    args.out_root.mkdir(parents=True, exist_ok=True)
    write_plan(args, graph_grid, client_grid)
    if args.phase == "plan":
        return
    try:
        preflight(args)
    except Exception as exc:
        if args.phase == "preflight" or not args.allow_no_gpu:
            raise SystemExit(str(exc)) from exc
        print(f"preflight warning: {exc}", file=sys.stderr)
    if args.phase == "preflight":
        return
    if args.phase in ("all", "benchmark"):
        index = 0
        for mode in selected_graph_modes(args):
            for backend_name in selected_backends(args):
                print(f"[group] {mode} {backend_name}", flush=True)
                run_group(
                    args,
                    mode=mode,
                    backend_name=backend_name,
                    graph_grid=graph_grid,
                    client_grid=client_grid,
                    group_index=index,
                )
                index += 1
    if args.phase in ("all", "plot"):
        plot_and_summarize(args, graph_grid)


if __name__ == "__main__":
    main()
