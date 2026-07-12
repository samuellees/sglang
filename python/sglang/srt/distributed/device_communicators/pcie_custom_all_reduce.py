from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils.cpp_extension import load

from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    is_weak_contiguous,
)
from sglang.srt.distributed.device_communicators.pcie_ar_config import (
    ArShape,
    PolicyStore,
    default_config_dir,
    normalize_dtype,
)
from sglang.srt.environ import envs
from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)


def _source_path() -> Path:
    configured = envs.SGLANG_PCIE_AR_SOURCE_PATH.get()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent / "pcie_ar" / "symm_allreduce_ext.cu"


def _build_dir() -> Path:
    configured = envs.SGLANG_PCIE_AR_BUILD_DIR.get()
    if configured:
        return Path(configured)
    return Path(envs.SGLANG_CACHE_DIR.get()) / "pcie_allreduce" / "build"


def _dtype_name(tensor: torch.Tensor) -> str:
    if tensor.dtype is torch.bfloat16:
        return "bf16"
    if tensor.dtype is torch.float16:
        return "fp16"
    if tensor.dtype is torch.float32:
        return "fp32"
    return normalize_dtype(str(tensor.dtype))


class PcieCustomAllReduce:
    _SUPPORTED_WORLD_SIZES = (2, 4, 8)

    def __init__(self, group: ProcessGroup, device: torch.device) -> None:
        self.disabled = True
        self.group = group
        self.device = device
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self.max_size = int(envs.SGLANG_PCIE_AR_MAX_SIZE_BYTES.get())
        self.max_blocks = int(envs.SGLANG_PCIE_AR_MAX_BLOCKS.get())
        self.enable_pdl = bool(envs.SGLANG_PCIE_AR_ENABLE_PDL.get())
        self._IS_CAPTURING = False
        self.ext: Any | None = None
        self.obj: Any | None = None

        if self.world_size not in self._SUPPORTED_WORLD_SIZES:
            return

        self.policy_store = PolicyStore.load(envs.SGLANG_PCIE_AR_CONFIG_DIR.get())
        self.policy_store = self.policy_store.filter_tp(self.world_size)
        if not self.policy_store.rows:
            log_info_on_rank0(
                logger,
                "[AR] PCIe custom allreduce disabled: no policy rows found under "
                f"{default_config_dir()} for TP{self.world_size}.",
            )
            return

        if len(self.policy_store.dtypes) != 1:
            log_info_on_rank0(
                logger,
                "[AR] PCIe custom allreduce disabled: first integration supports "
                f"one dtype per process group, got {sorted(self.policy_store.dtypes)}.",
            )
            return

        source = _source_path()
        if not source.exists():
            log_info_on_rank0(
                logger,
                f"[AR] PCIe custom allreduce disabled: source not found at {source}.",
            )
            return

        dtype = next(iter(self.policy_store.dtypes))
        elem_size = torch.empty((), dtype=self._torch_dtype(dtype)).element_size()
        max_numel = max(self.policy_store.max_numel, self.max_size // elem_size)
        self.max_blocks = max(self.max_blocks, self.policy_store.max_blocks)
        self.ext = self._build_extension(source)
        self.obj = self.ext.IpcPushAllreduce(
            self.rank,
            self.world_size,
            max_numel,
            elem_size,
            self.max_blocks,
        )
        handles: list[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(handles, self.obj.share_storage(), group=self.group)
        self.obj.post_init(handles)
        dist.barrier(group=self.group)
        self.disabled = False
        log_info_on_rank0(
            logger,
            "[AR] PCIe custom allreduce initialized: "
            f"tp={self.world_size}, rows={len(self.policy_store.rows)}, "
            f"config_dir={default_config_dir()}, pdl={self.enable_pdl}.",
        )

    @staticmethod
    def _torch_dtype(dtype: str) -> torch.dtype:
        dtype = normalize_dtype(dtype)
        if dtype == "bf16":
            return torch.bfloat16
        if dtype == "fp16":
            return torch.float16
        if dtype == "fp32":
            return torch.float32
        raise ValueError(f"unsupported PCIe allreduce dtype: {dtype}")

    def _build_extension(self, source: Path) -> Any:
        build_dir = _build_dir()
        build_dir.mkdir(parents=True, exist_ok=True)
        return load(
            name="sglang_pcie_allreduce_ext",
            sources=[str(source)],
            build_directory=str(build_dir),
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=bool(envs.SGLANG_PCIE_AR_VERBOSE_BUILD.get()),
        )

    @contextmanager
    def capture(self):
        self._IS_CAPTURING = True
        try:
            yield
        finally:
            self._IS_CAPTURING = False

    def _shape_for(self, inp: torch.Tensor) -> ArShape | None:
        if inp.ndim == 0:
            return None
        hidden = int(inp.shape[-1])
        if hidden <= 0:
            return None
        batch = int(inp.numel() // hidden)
        if batch * hidden != int(inp.numel()):
            return None
        return ArShape(self.world_size, hidden, batch, _dtype_name(inp))

    def _policy_for(self, inp: torch.Tensor):
        shape = self._shape_for(inp)
        if shape is None:
            return None
        return self.policy_store.find(shape)

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        if not inp.is_cuda:
            return False
        input_bytes = inp.numel() * inp.element_size()
        if input_bytes > self.max_size or input_bytes % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        return self._policy_for(inp) is not None

    def _all_reduce_impl(self, input: torch.Tensor, policy) -> torch.Tensor:
        assert self.obj is not None
        out = torch.empty_like(input)
        stream_mode = policy.algorithm.endswith("_stream")
        self.obj.all_reduce_v2(
            input,
            out,
            input.numel(),
            policy.blocks,
            policy.threads,
            stream_mode,
            self.enable_pdl,
            self.enable_pdl,
        )
        return out

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor | None:
        if self.disabled or not self.should_custom_ar(input):
            return None
        policy = self._policy_for(input)
        assert policy is not None
        if self._IS_CAPTURING and not torch.cuda.is_current_stream_capturing():
            if is_in_piecewise_cuda_graph():
                return self._all_reduce_impl(input, policy)
            return torch.zeros_like(input)
        return self._all_reduce_impl(input, policy)

    def close(self) -> None:
        if self.obj is not None:
            self.obj.close()
            self.obj = None

    def __del__(self):
        self.close()
