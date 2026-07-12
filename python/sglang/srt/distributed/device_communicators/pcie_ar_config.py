from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BATCH_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
FINE_BATCH_GRID = (
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
)

DTYPE_ALIASES = {
    "torch.bfloat16": "bf16",
    "bfloat16": "bf16",
    "bf16": "bf16",
    "torch.float16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "fp16": "fp16",
    "torch.float32": "fp32",
    "float32": "fp32",
    "fp32": "fp32",
}

DTYPE_SIZE = {
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
}


def normalize_dtype(dtype: Any) -> str:
    return DTYPE_ALIASES.get(str(dtype).lower(), str(dtype).lower())


def default_config_dir() -> Path:
    root = os.environ.get("SGLANG_PCIE_AR_CONFIG_DIR")
    if root:
        return Path(root)
    return Path(__file__).resolve().parent / "pcie_ar_configs"


def sanitize_device_name(device_name: str) -> str:
    return device_name.replace(" ", "_").replace("/", "_")


def policy_path_for_shape(
    config_dir: str | Path,
    device_name: str,
    tp: int,
    hidden: int,
    dtype: str,
    profile: str | None = None,
) -> Path:
    root = Path(config_dir)
    if profile:
        return root / "profiles" / profile / "policy.json"
    device = sanitize_device_name(device_name)
    return root / device / f"tp{tp}" / f"h{hidden}" / normalize_dtype(dtype) / "policy.json"


@dataclass(frozen=True)
class ArShape:
    tp: int
    hidden: int
    batch: int
    dtype: str = "bf16"

    @property
    def key(self) -> tuple[int, int, int, str]:
        return self.tp, self.hidden, self.batch, normalize_dtype(self.dtype)


@dataclass(frozen=True)
class LaunchPolicy:
    tp: int
    hidden: int
    batch: int
    dtype: str
    algorithm: str
    blocks: int
    threads: int
    vec_packs: int = 1
    latency_us: float | None = None
    comment: str = ""

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "LaunchPolicy":
        latency = row.get("latency_us")
        return cls(
            tp=int(row["tp"]),
            hidden=int(row["hidden"]),
            batch=int(row["batch"]),
            dtype=normalize_dtype(row.get("dtype", "bf16")),
            algorithm=str(row["algorithm"]),
            blocks=int(row["blocks"]),
            threads=int(row["threads"]),
            vec_packs=int(row.get("vec_packs", 1)),
            latency_us=None if latency in (None, "") else float(latency),
            comment=str(row.get("comment", "")),
        )

    @property
    def shape(self) -> ArShape:
        return ArShape(self.tp, self.hidden, self.batch, self.dtype)

    def to_json(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "tp": self.tp,
            "hidden": self.hidden,
            "dtype": self.dtype,
            "batch": self.batch,
            "algorithm": self.algorithm,
            "blocks": self.blocks,
            "threads": self.threads,
            "vec_packs": self.vec_packs,
        }
        if self.latency_us is not None:
            row["latency_us"] = self.latency_us
        if self.comment:
            row["comment"] = self.comment
        return row


class PolicyStore:
    def __init__(self, rows: Iterable[LaunchPolicy], paths: Iterable[Path]):
        self.rows = list(rows)
        self.paths = list(paths)
        self._by_shape = {row.shape.key: row for row in self.rows}

    @classmethod
    def load(cls, config_dir: str | Path | None = None) -> "PolicyStore":
        root = Path(config_dir) if config_dir else default_config_dir()
        rows: list[LaunchPolicy] = []
        paths: list[Path] = []
        if root.exists():
            for path in sorted(root.rglob("policy.json")):
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if int(data.get("schema_version", 1)) != 1:
                    continue
                paths.append(path)
                rows.extend(LaunchPolicy.from_json(row) for row in data.get("rows", []))
        return cls(rows, paths)

    def filter_tp(self, tp: int) -> "PolicyStore":
        return PolicyStore((row for row in self.rows if row.tp == tp), self.paths)

    def find(self, shape: ArShape) -> LaunchPolicy | None:
        return self._by_shape.get(shape.key)

    @property
    def max_blocks(self) -> int:
        return max((row.blocks for row in self.rows), default=1)

    @property
    def max_numel(self) -> int:
        return max((row.hidden * row.batch for row in self.rows), default=0)

    @property
    def dtypes(self) -> set[str]:
        return {row.dtype for row in self.rows}


def write_policy(path: str | Path, profile_name: str, rows: list[LaunchPolicy]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "profile_name": profile_name,
        "rows": [row.to_json() for row in rows],
        "fallback": "nccl",
    }
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def parse_batch_grid(raw: str | None) -> list[int]:
    if raw is None:
        return list(DEFAULT_BATCH_GRID)

    batches: list[int] = []
    for item in raw.replace(" ", "").split(","):
        if not item:
            continue
        if item == "default":
            batches.extend(DEFAULT_BATCH_GRID)
        elif item == "fine":
            batches.extend(FINE_BATCH_GRID)
        else:
            batches.append(int(item))
    return sorted(set(batches))


def _ipc_v2_algorithm(stream_mode: bool) -> str:
    return "custom_ipc_v2_remote_push_stream" if stream_mode else "custom_ipc_v2_remote_push"


def heuristic_policy(tp: int, hidden: int, batch: int, max_blocks: int) -> tuple[str, int, int]:
    if tp == 2 and hidden <= 2048:
        stream_mode = False
        if batch <= 1:
            return _ipc_v2_algorithm(stream_mode), min(32, max_blocks), 64
        if batch <= 2:
            return _ipc_v2_algorithm(stream_mode), min(128, max_blocks), 64
        if batch <= 4:
            return _ipc_v2_algorithm(True), min(64, max_blocks), 128
        if batch <= 8:
            return _ipc_v2_algorithm(True), min(96, max_blocks), 64
        if batch <= 12:
            return _ipc_v2_algorithm(stream_mode), min(16, max_blocks), 64
        if batch <= 16:
            return _ipc_v2_algorithm(stream_mode), min(64, max_blocks), 128
        if batch <= 28:
            return _ipc_v2_algorithm(stream_mode), min(16, max_blocks), 64
        if batch <= 32:
            return _ipc_v2_algorithm(True), min(64, max_blocks), 64
        if batch <= 44:
            return _ipc_v2_algorithm(stream_mode), min(16, max_blocks), 128
        if batch <= 48:
            return _ipc_v2_algorithm(True), min(32, max_blocks), 128
        return _ipc_v2_algorithm(stream_mode), min(16, max_blocks), 128
    if tp == 4 and hidden == 4096:
        if batch <= 1:
            return _ipc_v2_algorithm(False), min(8, max_blocks), 128
        if batch <= 4:
            return _ipc_v2_algorithm(False), min(64, max_blocks), 128
        if batch <= 8:
            return _ipc_v2_algorithm(False), min(96, max_blocks), 128
        if batch <= 40:
            return _ipc_v2_algorithm(batch > 16), min(16, max_blocks), 128
        return _ipc_v2_algorithm(True), min(32, max_blocks), 128
    if tp == 8 and hidden >= 6144:
        if batch <= 4:
            return _ipc_v2_algorithm(False), min(12, max_blocks), 256
        if batch <= 32:
            return _ipc_v2_algorithm(True), min(32, max_blocks), 128
        return _ipc_v2_algorithm(True), min(8, max_blocks), 512
    return _ipc_v2_algorithm(False), min(16, max_blocks), 128


def generate_heuristic_rows(
    tp: int,
    hidden: int,
    dtype: str,
    batch_grid: Iterable[int],
    max_blocks: int,
    max_bytes: int,
) -> list[LaunchPolicy]:
    dtype = normalize_dtype(dtype)
    elem_size = DTYPE_SIZE[dtype]
    rows: list[LaunchPolicy] = []
    for batch in batch_grid:
        if batch * hidden * elem_size > max_bytes:
            continue
        algorithm, blocks, threads = heuristic_policy(tp, hidden, batch, max_blocks)
        rows.append(
            LaunchPolicy(
                tp=tp,
                hidden=hidden,
                batch=batch,
                dtype=dtype,
                algorithm=algorithm,
                blocks=blocks,
                threads=threads,
                comment="heuristic_seed",
            )
        )
    return rows
