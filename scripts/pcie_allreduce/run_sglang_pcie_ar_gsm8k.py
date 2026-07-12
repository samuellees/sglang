#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SGLANG_ROOT = Path(__file__).resolve().parents[2]
HARNESS_ROOT = SGLANG_ROOT.parent
DEFAULT_PYTHON = HARNESS_ROOT / ".pixi" / "envs" / "vllm" / "bin" / "python"
DEFAULT_MODEL = os.environ.get("SGLANG_PCIE_AR_MODEL_PATH", "")
DEFAULT_OUT = (
    HARNESS_ROOT.parent
    / "task-output"
    / "sglang-pcie-ar-qwen35-35b-tp2-gsm8k-20260711"
)


@dataclass(frozen=True)
class Case:
    name: str
    use_pcie_ar: bool
    pdl: bool


CASES = (
    Case("baseline", False, False),
    Case("ipc-pdl-off", True, False),
    Case("ipc-pdl-on", True, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GSM8K lm-eval against SGLang TP2 baseline and PCIe IPC allreduce."
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--sglang-root", type=Path, default=SGLANG_ROOT)
    parser.add_argument("--harness-root", type=Path, default=HARNESS_ROOT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--served-model-name", default="qwen35-35b")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--gpus", default="4,5")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=35200)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--max-running-requests", type=int, default=16)
    parser.add_argument("--cuda-graph-bs", default="16")
    parser.add_argument("--chunked-prefill-size", type=int, default=4096)
    parser.add_argument("--max-gen-toks", type=int, default=10000)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--num-concurrent", type=int, default=16)
    parser.add_argument("--num-fewshot", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ready-timeout-seconds", type=int, default=1800)
    parser.add_argument("--eval-timeout-seconds", type=int, default=7200)
    parser.add_argument("--server-exit-timeout-seconds", type=int, default=120)
    parser.add_argument("--only", default="")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--extra-server-arg", action="append", default=[])
    return parser.parse_args()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def free_port(start: int) -> int:
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port found from {start}")


def http_get_json(url: str, timeout: float = 10.0) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.getcode() == 200, resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def selected_cases(filters: str) -> list[Case]:
    parts = [part for part in filters.split(",") if part]
    if not parts:
        return list(CASES)
    return [case for case in CASES if any(part in case.name for part in parts)]


def validate_model_path(args: argparse.Namespace) -> None:
    if not args.model_path:
        raise SystemExit(
            "model path is required; pass --model-path or set SGLANG_PCIE_AR_MODEL_PATH."
        )
    if not Path(args.model_path).exists():
        raise SystemExit(f"model path does not exist: {args.model_path}")


def build_env(args: argparse.Namespace, case: Case) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str(args.sglang_root / "python")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    cache_root = args.out_root / "cache"
    pixi_bin = args.harness_root.parent / ".tools" / "pixi" / "bin" / "pixi"
    pixi_path = str(pixi_bin.parent)
    path = env.get("PATH", "")
    env.update(
        {
            "PYTHONPATH": ":".join(pythonpath),
            "PATH": f"{pixi_path}:{path}" if path else pixi_path,
            "PIXI_BIN": str(pixi_bin),
            "CUDA_VISIBLE_DEVICES": args.gpus,
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "FLASHINFER_CACHE_DIR": str(cache_root / "flashinfer"),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
            "HOME": str(cache_root / "home"),
            "NCCL_DEBUG": env.get("NCCL_DEBUG", "WARN"),
            "TORCH_CUDA_ARCH_LIST": env.get("TORCH_CUDA_ARCH_LIST", "12.0"),
            "SGLANG_PROFILE_V2": "0",
            "SGLANG_PCIE_AR_BUILD_DIR": str(cache_root / "pcie_ar_build"),
            "SGLANG_PCIE_AR_CONFIG_DIR": str(
                args.sglang_root
                / "python"
                / "sglang"
                / "srt"
                / "distributed"
                / "device_communicators"
                / "pcie_ar_configs"
            ),
            "SGLANG_PCIE_AR_ENABLE_PDL": "1" if case.pdl else "0",
            "SGLANG_PCIE_AR_MAX_SIZE_BYTES": str(8 * 1024 * 1024),
            "SGLANG_USE_PCIE_CUSTOM_ALL_REDUCE": "1" if case.use_pcie_ar else "0",
        }
    )
    if case.use_pcie_ar:
        env["SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"] = "0"
    else:
        env.pop("SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2", None)
    return env


def server_command(args: argparse.Namespace, port: int) -> list[str]:
    cmd = [
        str(args.python),
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(port),
        "--trust-remote-code",
        "--tensor-parallel-size",
        str(args.tp),
        "--context-length",
        str(args.context_length),
        "--max-running-requests",
        str(args.max_running_requests),
        "--cuda-graph-bs",
        *[item for item in args.cuda_graph_bs.replace(",", " ").split() if item],
        "--chunked-prefill-size",
        str(args.chunked_prefill_size),
        "--log-level",
        "info",
    ]
    cmd.extend(args.extra_server_arg)
    return cmd


def eval_command(args: argparse.Namespace, port: int, case_dir: Path) -> list[str]:
    cmd = [
        str(args.python),
        "tools/eval_lm_eval.py",
        "--host",
        args.host,
        "--port",
        str(port),
        "--served-model-name",
        args.served_model_name,
        "--engine",
        "lm_eval",
        "--suite",
        "gsm8k",
        "--task",
        "gsm8k",
        "--backend",
        "local-completions",
        "--primary-metric",
        "exact_match,flexible-extract",
        "--seed",
        str(args.seed),
        "--batch-size",
        "1",
        "--max-gen-toks",
        str(args.max_gen_toks),
        "--max-length",
        str(args.max_length),
        "--min-score",
        "0.0",
        "--num-concurrent",
        str(args.num_concurrent),
        "--num-fewshot",
        str(args.num_fewshot),
        "--timeout",
        str(args.eval_timeout_seconds),
        "--tokenizer-backend",
        "none",
        "--tokenized-requests",
        "false",
        "--run-dir",
        str(case_dir / "eval"),
        "--server-manifest",
        str(case_dir / "server" / "server.json"),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    return cmd


def wait_ready(base_url: str, timeout_s: int, proc: subprocess.Popen[Any]) -> None:
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        ok, body = http_get_json(f"{base_url}/v1/models", timeout=10)
        if ok:
            return
        last = body[:500]
        time.sleep(2)
    raise TimeoutError(f"server did not become ready: {last}")


def terminate_process(proc: subprocess.Popen[Any], timeout_s: int) -> int:
    if proc.poll() is not None:
        return int(proc.returncode)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return int(proc.wait(timeout=5))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return int(proc.returncode)
        time.sleep(1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return int(proc.wait(timeout=30))


def summarize_case(case_dir: Path) -> dict[str, Any]:
    result_path = case_dir / "eval" / "results.json"
    eval_path = case_dir / "eval" / "eval.json"
    payload: dict[str, Any] = {
        "results_json": str(result_path),
        "eval_json": str(eval_path),
        "result_exists": result_path.exists(),
    }
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        payload.update(
            {
                "primary_metric": result.get("primary_metric"),
                "primary_score": result.get("primary_score"),
                "passed": result.get("passed"),
                "metrics": result.get("metrics"),
            }
        )
    if eval_path.exists():
        meta = json.loads(eval_path.read_text(encoding="utf-8"))
        payload["returncode"] = meta.get("returncode")
        payload["duration_seconds"] = meta.get("duration_seconds")
        if meta.get("error"):
            payload["error"] = meta["error"]
    return payload


def run_case(args: argparse.Namespace, case: Case, index: int) -> dict[str, Any]:
    case_dir = args.out_root / "cases" / f"qwen35-35b-fp8-tp2-gsm8k-10k-{case.name}"
    if args.skip_existing and (case_dir / "eval" / "results.json").exists():
        summary = summarize_case(case_dir)
        summary.update({"case": case.name, "status": "skipped"})
        return summary

    port = free_port(args.base_port + index * 10)
    base_url = f"http://{args.host}:{port}"
    (case_dir / "server").mkdir(parents=True, exist_ok=True)
    command = server_command(args, port)
    env = build_env(args, case)
    write_json(
        case_dir / "server" / "server.json",
        {
            "created_at": now(),
            "case": case.name,
            "ready_url": f"{base_url}/v1/models",
            "command": command,
            "env_overrides": {
                key: env[key]
                for key in sorted(env)
                if key.startswith("SGLANG_")
                or key in {"CUDA_VISIBLE_DEVICES", "PYTHONPATH", "TORCH_CUDA_ARCH_LIST"}
            },
        },
    )
    with (case_dir / "server" / "server.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            cwd=str(args.sglang_root),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            text=True,
        )
    try:
        wait_ready(base_url, args.ready_timeout_seconds, proc)
        eval_cmd = eval_command(args, port, case_dir)
        write_json(
            case_dir / "eval-command.json",
            {
                "created_at": now(),
                "case": case.name,
                "command": eval_cmd,
                "cwd": str(args.harness_root),
            },
        )
        started = time.monotonic()
        eval_proc = subprocess.run(
            eval_cmd,
            cwd=str(args.harness_root),
            env=env,
            timeout=args.eval_timeout_seconds + 600,
            text=True,
        )
        duration = time.monotonic() - started
        summary = summarize_case(case_dir)
        summary.update(
            {
                "case": case.name,
                "status": "passed" if eval_proc.returncode == 0 else "failed",
                "eval_returncode": eval_proc.returncode,
                "wall_seconds": round(duration, 3),
            }
        )
        if eval_proc.returncode != 0 and "error" not in summary:
            summary["error"] = f"eval command exited with {eval_proc.returncode}"
        return summary
    except Exception as exc:
        return {
            "case": case.name,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        rc = terminate_process(proc, args.server_exit_timeout_seconds)
        server_json = case_dir / "server" / "server.json"
        if server_json.exists():
            data = json.loads(server_json.read_text(encoding="utf-8"))
            data["finished_at"] = now()
            data["server_returncode"] = rc
            write_json(server_json, data)


def write_summary(out_root: Path, rows: list[dict[str, Any]]) -> None:
    write_json(out_root / "summary.json", {"created_at": now(), "cases": rows})
    lines = [
        "case,status,score,metric,duration_seconds,wall_seconds,results_json,eval_json,error"
    ]
    for row in rows:
        lines.append(
            ",".join(
                [
                    str(row.get("case", "")),
                    str(row.get("status", "")),
                    "" if row.get("primary_score") is None else str(row["primary_score"]),
                    str(row.get("primary_metric", "")),
                    str(row.get("duration_seconds", "")),
                    str(row.get("wall_seconds", "")),
                    str(row.get("results_json", "")),
                    str(row.get("eval_json", "")),
                    json.dumps(str(row.get("error", "")), ensure_ascii=False),
                ]
            )
        )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "created_at": now(),
        "model": args.model_path,
        "served_model_name": args.served_model_name,
        "tp": args.tp,
        "gpus": args.gpus,
        "gsm8k": {
            "max_gen_toks": args.max_gen_toks,
            "max_length": args.max_length,
            "num_concurrent": args.num_concurrent,
            "num_fewshot": args.num_fewshot,
            "seed": args.seed,
            "limit": args.limit,
        },
        "cases": [case.name for case in selected_cases(args.only)],
    }
    write_json(args.out_root / "plan.json", plan)
    validate_model_path(args)
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(selected_cases(args.only)):
        print(f"[gsm8k] {case.name}", flush=True)
        row = run_case(args, case, index)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        rows.append(row)
        write_summary(args.out_root, rows)
    return 0 if all(row.get("status") in {"passed", "skipped"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
