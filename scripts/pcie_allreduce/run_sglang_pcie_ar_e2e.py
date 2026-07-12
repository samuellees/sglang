#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = (
    REPO_ROOT.parent / ".pixi" / "envs" / "vllm" / "bin" / "python"
)
DEFAULT_MODEL = os.environ.get("SGLANG_PCIE_AR_MODEL_PATH", "")
DEFAULT_NSYS = (
    REPO_ROOT.parent
    / ".pixi"
    / "envs"
    / "vllm"
    / "nsight-compute-2026.1.1"
    / "host"
    / "target-linux-x64"
    / "nsys"
)
DEFAULT_OUT = (
    REPO_ROOT.parent.parent
    / "task-output"
    / "sglang-pcie-ar-qwen35-35b-tp2-ifb-20260711"
)


@dataclass(frozen=True)
class Backend:
    name: str
    use_pcie_ar: bool
    pdl: bool


@dataclass(frozen=True)
class GraphCase:
    label: str
    concurrency: int
    cuda_graph_bs: tuple[int, ...]


@dataclass(frozen=True)
class Case:
    backend: Backend
    graph: GraphCase

    @property
    def name(self) -> str:
        if self.backend.name == "baseline":
            backend = "baseline"
        else:
            backend = "ipc-pdl-on" if self.backend.pdl else "ipc-pdl-off"
        graph = "-".join(str(v) for v in self.graph.cuda_graph_bs)
        return (
            f"qwen35-35b-fp8-tp2-ifb-cc{self.graph.concurrency}"
            f"-cg{graph}-{self.graph.label}-{backend}"
        )


BACKENDS = (
    Backend("baseline", use_pcie_ar=False, pdl=False),
    Backend("ipc", use_pcie_ar=True, pdl=False),
    Backend("ipc_pdl", use_pcie_ar=True, pdl=True),
)

GRAPH_CASES = (
    GraphCase("bs16", 16, (16,)),
    GraphCase("bs14-padded-to-16", 14, (16,)),
    GraphCase("bs14-exact", 14, (14, 16)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen3.5-35B TP2 SGLang baseline vs PCIe IPC allreduce "
            "end-to-end benchmarks and 5-step decode nsys captures."
        )
    )
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--sglang-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--served-model-name", default="qwen35-35b")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument(
        "--gpus",
        default="4,5",
        help="CUDA_VISIBLE_DEVICES for TP2. Defaults to the later GPUs.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=33100)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--nsys-bin", type=Path, default=DEFAULT_NSYS)
    parser.add_argument(
        "--phase",
        choices=("all", "benchmark", "nsys", "plan", "preflight"),
        default="all",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated substring filters matched against case names.",
    )
    parser.add_argument("--num-prompts", type=int, default=160)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--profile-wait-seconds", type=float, default=20.0)
    parser.add_argument("--ready-timeout-seconds", type=int, default=1800)
    parser.add_argument("--bench-timeout-seconds", type=int, default=2400)
    parser.add_argument("--profile-timeout-seconds", type=int, default=900)
    parser.add_argument("--server-exit-timeout-seconds", type=int, default=120)
    parser.add_argument("--max-tokens-context", type=int, default=8192)
    parser.add_argument("--mem-fraction-static", default="")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--allow-no-gpu", action="store_true")
    parser.add_argument("--extra-server-arg", action="append", default=[])
    return parser.parse_args()


def all_cases() -> list[Case]:
    return [Case(backend, graph) for graph in GRAPH_CASES for backend in BACKENDS]


def selected_cases(filters: str) -> list[Case]:
    cases = all_cases()
    parts = [item for item in filters.split(",") if item]
    if not parts:
        return cases
    return [case for case in cases if any(part in case.name for part in parts)]


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def http_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 60.0,
) -> tuple[int, str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def free_port(start: int) -> int:
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port found from {start}")


def build_env(args: argparse.Namespace, case: Case) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [
        str(args.sglang_root / "python"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    cache_root = args.out_root / "cache"
    env.update(
        {
            "PYTHONPATH": ":".join(pythonpath),
            "CUDA_VISIBLE_DEVICES": args.gpus,
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "FLASHINFER_CACHE_DIR": str(cache_root / "flashinfer"),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
            "HOME": str(cache_root / "home"),
            "NCCL_DEBUG": env.get("NCCL_DEBUG", "WARN"),
            "TORCH_CUDA_ARCH_LIST": env.get("TORCH_CUDA_ARCH_LIST", "12.0"),
            "SGLANG_PROFILE_V2": "0",
            "SGLANG_PCIE_AR_BUILD_DIR": str(cache_root / "pcie_ar_build"),
            "SGLANG_PCIE_AR_CONFIG_DIR": env.get(
                "SGLANG_PCIE_AR_CONFIG_DIR",
                str(
                    args.sglang_root
                    / "python"
                    / "sglang"
                    / "srt"
                    / "distributed"
                    / "device_communicators"
                    / "pcie_ar_configs"
                ),
            ),
            "SGLANG_PCIE_AR_ENABLE_PDL": "1" if case.backend.pdl else "0",
            "SGLANG_PCIE_AR_MAX_SIZE_BYTES": str(8 * 1024 * 1024),
            "SGLANG_USE_PCIE_CUSTOM_ALL_REDUCE": (
                "1" if case.backend.use_pcie_ar else "0"
            ),
        }
    )
    if case.backend.use_pcie_ar:
        env["SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"] = "0"
    else:
        env.pop("SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2", None)
    return env


def server_command(args: argparse.Namespace, case: Case, port: int) -> list[str]:
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
        str(args.max_tokens_context),
        "--max-running-requests",
        str(case.graph.concurrency),
        "--cuda-graph-bs",
        *[str(v) for v in case.graph.cuda_graph_bs],
        "--chunked-prefill-size",
        "4096",
        "--log-level",
        "info",
    ]
    if args.mem_fraction_static:
        cmd.extend(["--mem-fraction-static", args.mem_fraction_static])
    cmd.extend(args.extra_server_arg)
    return cmd


def nsys_command(
    args: argparse.Namespace, case: Case, port: int, case_dir: Path
) -> list[str]:
    output_base = case_dir / "nsys" / f"{case.name}-decode-5steps"
    return [
        str(args.nsys_bin),
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,nvtx,osrt,cudnn,cublas",
        "--trace-fork-before-exec=true",
        "--sample=none",
        "--cpuctxsw=none",
        "--cuda-graph-trace=node",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "-o",
        str(output_base),
        *server_command(args, case, port),
    ]


def wait_ready(base_url: str, timeout_s: int, proc: subprocess.Popen[Any]) -> None:
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            status, body = http_json(f"{base_url}/v1/models", timeout=10)
            if status == 200:
                return
            last = f"status={status} body={body[:200]}"
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(2)
    raise TimeoutError(f"server did not become ready: {last}")


def launch_server(
    args: argparse.Namespace,
    case: Case,
    case_dir: Path,
    *,
    profile: bool,
    port: int,
) -> tuple[subprocess.Popen[Any], list[str], dict[str, str]]:
    env = build_env(args, case)
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "server").mkdir(exist_ok=True)
    (case_dir / "nsys").mkdir(exist_ok=True)
    cmd = nsys_command(args, case, port, case_dir) if profile else server_command(args, case, port)
    log_path = case_dir / "server" / ("server.nsys.log" if profile else "server.log")
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(args.sglang_root),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
    )
    return proc, cmd, env


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


def prompt_text(input_len: int, seed: int) -> str:
    # This is intentionally dependency-free; exact token count is close enough for
    # serving comparisons and avoids importing tokenizer/model code in the client.
    vocab = (
        "the",
        "and",
        "of",
        "to",
        "in",
        "that",
        "with",
        "for",
        "as",
        "is",
        "on",
        "by",
        "from",
        "this",
        "it",
        "be",
    )
    words = [vocab[(seed + i) % len(vocab)] for i in range(input_len)]
    return " ".join(words)


def parse_stream_response(
    base_url: str,
    model: str,
    prompt: str,
    output_len: int,
    timeout_s: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_len,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first = None
    chunks = 0
    text_bytes = 0
    error = None
    status = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = resp.getcode()
            while True:
                line = resp.readline()
                if not line:
                    break
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                if first is None:
                    first = time.perf_counter()
                chunks += 1
                text_bytes += len(payload)
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", errors="replace")
        error = f"HTTPError: HTTP Error {exc.code}: {body[:1000]}"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    end = time.perf_counter()
    ttft_ms = None if first is None else (first - start) * 1000.0
    tpot_ms = None
    if first is not None and chunks > 1:
        tpot_ms = (end - first) * 1000.0 / (chunks - 1)
    return {
        "status": status,
        "error": error,
        "chunks": chunks,
        "text_bytes": text_bytes,
        "start_perf": start,
        "first_perf": first,
        "end_perf": end,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": (end - start) * 1000.0,
    }


async def run_client(
    base_url: str,
    *,
    model: str,
    concurrency: int,
    num_prompts: int,
    input_len: int,
    output_len: int,
    timeout_s: int,
    out_dir: Path,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=concurrency)
    sem = asyncio.Semaphore(concurrency)
    request_path = out_dir / "requests.jsonl"
    if request_path.exists():
        request_path.unlink()
    started = time.perf_counter()

    async def one(index: int) -> dict[str, Any]:
        async with sem:
            prompt = prompt_text(input_len, index)
            result = await loop.run_in_executor(
                executor,
                parse_stream_response,
                base_url,
                model,
                prompt,
                output_len,
                timeout_s,
            )
            result["index"] = index
            append_jsonl(request_path, result)
            return result

    tasks = [asyncio.create_task(one(i)) for i in range(num_prompts)]
    results = await asyncio.wait_for(
        asyncio.gather(*tasks),
        timeout=max(timeout_s, timeout_s * math.ceil(num_prompts / concurrency)),
    )
    finished = time.perf_counter()
    executor.shutdown(wait=False, cancel_futures=True)
    good = [item for item in results if item.get("error") is None and item["chunks"] > 0]
    ttfts = sorted(item["ttft_ms"] for item in good if item["ttft_ms"] is not None)
    tpots = sorted(item["tpot_ms"] for item in good if item["tpot_ms"] is not None)
    e2es = sorted(item["e2e_ms"] for item in good)
    total_chunks = sum(item["chunks"] for item in good)
    duration_s = finished - started
    summary = {
        "created_at": now(),
        "base_url": base_url,
        "model": model,
        "concurrency": concurrency,
        "num_prompts": num_prompts,
        "input_len_target": input_len,
        "output_len_target": output_len,
        "duration_s": duration_s,
        "ok_requests": len(good),
        "failed_requests": len(results) - len(good),
        "total_stream_chunks": total_chunks,
        "stream_chunks_per_s": total_chunks / duration_s if duration_s > 0 else None,
        "ttft_ms_avg": mean(ttfts),
        "ttft_ms_p50": percentile(ttfts, 50),
        "ttft_ms_p90": percentile(ttfts, 90),
        "tpot_ms_avg": mean(tpots),
        "tpot_ms_p50": percentile(tpots, 50),
        "tpot_ms_p90": percentile(tpots, 90),
        "e2e_ms_avg": mean(e2es),
        "e2e_ms_p50": percentile(e2es, 50),
        "e2e_ms_p90": percentile(e2es, 90),
        "requests_jsonl": str(request_path),
    }
    write_json(out_dir / "benchmark.json", summary)
    return summary


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * (pct / 100.0)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - rank) + values[hi] * (rank - lo)


class StreamingRequest:
    def __init__(
        self,
        base_url: str,
        model: str,
        prompt: str,
        output_len: int,
        log_path: Path,
        timeout_s: int,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.prompt = prompt
        self.output_len = output_len
        self.log_path = log_path
        self.timeout_s = timeout_s
        self.first_token = asyncio.Event()
        self.done = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.error: str | None = None
        self._stop = False
        self._response = None

    def stop(self) -> None:
        self._stop = True
        if self._response is not None:
            try:
                self._response.close()
            except Exception:
                pass

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self._loop = loop
        await loop.run_in_executor(None, self._run_blocking)

    def _set_first_token(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.first_token.set)
        else:
            self.first_token.set()

    def _set_done(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.done.set)
        else:
            self.done.set()

    def _run_blocking(self) -> None:
        body = {
            "model": self.model,
            "prompt": self.prompt,
            "max_tokens": self.output_len,
            "temperature": 0,
            "stream": True,
            "ignore_eos": True,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                self._response = resp
                with self.log_path.open("w", encoding="utf-8") as f:
                    while not self._stop:
                        line = resp.readline()
                        if not line:
                            break
                        text = line.decode("utf-8", errors="replace")
                        f.write(text)
                        if text.startswith("data: ") and text.strip() != "data: [DONE]":
                            self._set_first_token()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            self.error = f"HTTPError: HTTP Error {exc.code}: {body[:1000]}"
            self._set_first_token()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._set_first_token()
        finally:
            self._set_done()


async def run_decode_profile(
    base_url: str,
    *,
    model: str,
    concurrency: int,
    input_len: int,
    output_len: int,
    profile_steps: int,
    wait_seconds: float,
    timeout_s: int,
    out_dir: Path,
    case_name: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    streams = [
        StreamingRequest(
            base_url,
            model,
            prompt_text(input_len, 10000 + i),
            output_len,
            out_dir / f"stream_{i:03d}.log",
            timeout_s,
        )
        for i in range(concurrency)
    ]
    tasks = [asyncio.create_task(item.run()) for item in streams]
    await asyncio.wait_for(
        asyncio.gather(*(item.first_token.wait() for item in streams)),
        timeout=timeout_s,
    )
    profile_body = {
        "output_dir": str(out_dir / "torch_profile"),
        "num_steps": profile_steps,
        "activities": ["CUDA_PROFILER"],
        "profile_by_stage": True,
        "profile_prefix": case_name,
        "profile_stages": ["decode"],
    }
    write_json(out_dir / "start_profile.request.json", profile_body)
    status, body = http_json(
        f"{base_url}/start_profile",
        payload=profile_body,
        timeout=timeout_s,
    )
    (out_dir / "start_profile.response.txt").write_text(body)
    if status != 200:
        for item in streams:
            item.stop()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise RuntimeError(f"/start_profile failed: status={status} body={body[:400]}")
    await asyncio.sleep(wait_seconds)
    for item in streams:
        item.stop()
    await asyncio.gather(*tasks, return_exceptions=True)
    manifest = {
        "created_at": now(),
        "base_url": base_url,
        "model": model,
        "concurrency": concurrency,
        "input_len_target": input_len,
        "output_len_target": output_len,
        "profile_steps": profile_steps,
        "wait_seconds": wait_seconds,
        "profile_body": profile_body,
        "stream_errors": [item.error for item in streams if item.error],
    }
    write_json(out_dir / "profile_client.json", manifest)
    return manifest


def run_benchmark_case(args: argparse.Namespace, case: Case, index: int) -> dict[str, Any]:
    case_dir = args.out_root / "cases" / case.name
    done = case_dir / "benchmark" / "benchmark.json"
    if args.skip_existing and done.exists():
        return {"case": case.name, "phase": "benchmark", "status": "skipped"}
    port = free_port(args.base_port + index * 10)
    base_url = f"http://{args.host}:{port}"
    proc, cmd, env = launch_server(args, case, case_dir, profile=False, port=port)
    manifest = {
        "case": case.name,
        "phase": "benchmark",
        "created_at": now(),
        "base_url": base_url,
        "server_command": cmd,
        "env_subset": env_subset(env),
    }
    write_json(case_dir / "benchmark" / "manifest.json", manifest)
    status = "failed"
    error = ""
    try:
        wait_ready(base_url, args.ready_timeout_seconds, proc)
        summary = asyncio.run(
            run_client(
                base_url,
                model=args.served_model_name,
                concurrency=case.graph.concurrency,
                num_prompts=args.num_prompts,
                input_len=args.input_len,
                output_len=args.output_len,
                timeout_s=args.bench_timeout_seconds,
                out_dir=case_dir / "benchmark",
            )
        )
        status = "passed" if summary["failed_requests"] == 0 else "failed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (case_dir / "benchmark" / "error.txt").write_text(error + "\n")
    finally:
        rc = terminate_process(proc, args.server_exit_timeout_seconds)
    result = {
        "case": case.name,
        "phase": "benchmark",
        "status": status,
        "error": error,
        "server_returncode": rc,
        "case_dir": str(case_dir),
    }
    append_jsonl(args.out_root / "results.jsonl", result)
    return result


def run_nsys_case(args: argparse.Namespace, case: Case, index: int) -> dict[str, Any]:
    case_dir = args.out_root / "cases" / case.name
    report = case_dir / "nsys" / f"{case.name}-decode-5steps.nsys-rep"
    if args.skip_existing and report.exists():
        return {"case": case.name, "phase": "nsys", "status": "skipped"}
    port = free_port(args.base_port + 1000 + index * 10)
    base_url = f"http://{args.host}:{port}"
    proc, cmd, env = launch_server(args, case, case_dir, profile=True, port=port)
    manifest = {
        "case": case.name,
        "phase": "nsys",
        "created_at": now(),
        "base_url": base_url,
        "server_command": cmd,
        "env_subset": env_subset(env),
        "expected_report": str(report),
    }
    write_json(case_dir / "profile" / "manifest.json", manifest)
    status = "failed"
    error = ""
    try:
        wait_ready(base_url, args.ready_timeout_seconds, proc)
        asyncio.run(
            run_decode_profile(
                base_url,
                model=args.served_model_name,
                concurrency=case.graph.concurrency,
                input_len=args.input_len,
                output_len=args.output_len,
                profile_steps=args.profile_steps,
                wait_seconds=args.profile_wait_seconds,
                timeout_s=args.profile_timeout_seconds,
                out_dir=case_dir / "profile",
                case_name=case.name,
            )
        )
        # Give nsys a moment to observe cudaProfilerStop before server teardown.
        time.sleep(5)
        status = "passed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (case_dir / "profile" / "error.txt").write_text(error + "\n")
    finally:
        rc = terminate_process(proc, args.server_exit_timeout_seconds)
    if status == "passed" and not report.exists():
        status = "failed"
        error = f"nsys report missing: {report}"
    result = {
        "case": case.name,
        "phase": "nsys",
        "status": status,
        "error": error,
        "server_returncode": rc,
        "case_dir": str(case_dir),
        "nsys_report": str(report),
    }
    append_jsonl(args.out_root / "results.jsonl", result)
    return result


def env_subset(env: dict[str, str]) -> dict[str, str]:
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "PYTHONPATH",
        "FLASHINFER_DISABLE_VERSION_CHECK",
        "FLASHINFER_CACHE_DIR",
        "XDG_CACHE_HOME",
        "HOME",
        "TORCH_CUDA_ARCH_LIST",
        "SGLANG_USE_PCIE_CUSTOM_ALL_REDUCE",
        "SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2",
        "SGLANG_PCIE_AR_ENABLE_PDL",
        "SGLANG_PCIE_AR_CONFIG_DIR",
        "SGLANG_PCIE_AR_BUILD_DIR",
        "SGLANG_PROFILE_V2",
        "NCCL_DEBUG",
    ]
    return {key: env[key] for key in keys if key in env}


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    checks = {
        "created_at": now(),
        "python": str(args.python),
        "python_exists": args.python.exists(),
        "sglang_root": str(args.sglang_root),
        "model_path": args.model_path,
        "model_path_exists": bool(args.model_path) and Path(args.model_path).exists(),
        "nsys_bin": str(args.nsys_bin),
        "nsys_exists": args.nsys_bin.exists(),
        "dev_nvidiactl_exists": Path("/dev/nvidiactl").exists(),
        "dev_nvidia0_exists": Path("/dev/nvidia0").exists(),
    }
    probe = (
        "import json, torch; "
        "print(json.dumps({'torch': torch.__version__, "
        "'cuda_available': torch.cuda.is_available(), "
        "'device_count': torch.cuda.device_count()}))"
    )
    try:
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
        raise RuntimeError("GPU device files are not visible; use --allow-no-gpu for plan/dry runs only.")
    return checks


def validate_model_path(args: argparse.Namespace) -> None:
    if not args.model_path:
        raise SystemExit(
            "model path is required for benchmark/nsys runs; pass --model-path "
            "or set SGLANG_PCIE_AR_MODEL_PATH."
        )
    if not Path(args.model_path).exists():
        raise SystemExit(f"model path does not exist: {args.model_path}")


def write_plan(args: argparse.Namespace, cases: list[Case]) -> None:
    rows = []
    for idx, case in enumerate(cases):
        port = args.base_port + idx * 10
        rows.append(
            {
                "case": case.name,
                "benchmark_port_hint": port,
                "nsys_port_hint": args.base_port + 1000 + idx * 10,
                "backend": case.backend.name,
                "pcie_ar": case.backend.use_pcie_ar,
                "pdl": case.backend.pdl,
                "concurrency": case.graph.concurrency,
                "cuda_graph_bs": list(case.graph.cuda_graph_bs),
                "case_dir": str(args.out_root / "cases" / case.name),
            }
        )
    write_json(args.out_root / "plan.json", {"created_at": now(), "cases": rows})
    print(f"plan: {args.out_root / 'plan.json'}")


def main() -> None:
    args = parse_args()
    cases = selected_cases(args.only)
    args.out_root.mkdir(parents=True, exist_ok=True)
    write_plan(args, cases)
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
    validate_model_path(args)
    for idx, case in enumerate(cases):
        if args.phase in ("all", "benchmark"):
            print(f"[benchmark] {case.name}", flush=True)
            print(json.dumps(run_benchmark_case(args, case, idx), ensure_ascii=False), flush=True)
        if args.phase in ("all", "nsys"):
            print(f"[nsys] {case.name}", flush=True)
            print(json.dumps(run_nsys_case(args, case, idx), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
