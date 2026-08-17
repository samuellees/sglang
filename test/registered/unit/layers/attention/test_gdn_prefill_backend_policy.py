import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, sentinel

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.linear import gdn_backend
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    GDNKernelDispatcher,
    flashinfer_gdn_prefill_default,
    should_split_gdn_mixed_prefill_decode,
)
from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    maybe_build_flashinfer_checkpoint_plan,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_runner(
    *,
    state_dtype=torch.bfloat16,
    key_dim=128,
    value_dim=128,
    **arg_overrides,
):
    args = SimpleNamespace(
        linear_attn_backend="triton",
        linear_attn_prefill_backend=None,
        uses_mamba_radix_cache=False,
        enable_page_major_kv_layout=False,
        mamba_radix_cache_strategy="no_buffer",
        enable_dynamic_chunking=False,
        chunked_prefill_size=8192,
    )
    for name, value in arg_overrides.items():
        setattr(args, name, value)

    return SimpleNamespace(
        server_args=args,
        model_config=SimpleNamespace(),
        hybrid_gdn_config=SimpleNamespace(
            linear_key_head_dim=key_dim,
            linear_value_head_dim=value_dim,
        ),
        req_to_token_pool=SimpleNamespace(
            mamba_pool=SimpleNamespace(
                mamba_cache=SimpleNamespace(temporal=SimpleNamespace(dtype=state_dtype))
            )
        ),
    )


class TestFlashInferGDNPrefillBackendPolicy(unittest.TestCase):
    def test_mixed_split_is_limited_to_validated_tp_execution(self):
        base = dict(
            is_target_verify=False,
            num_mixed_decode_reqs=2,
            has_mamba_track_mask=False,
        )
        self.assertTrue(
            should_split_gdn_mixed_prefill_decode(
                enable_dp_attention=False,
                **base,
            )
        )
        self.assertFalse(
            should_split_gdn_mixed_prefill_decode(
                enable_dp_attention=True,
                **base,
            )
        )

        for override in (
            {"is_target_verify": True},
            {"num_mixed_decode_reqs": 0},
            {"has_mamba_track_mask": True},
        ):
            with self.subTest(override=override):
                self.assertFalse(
                    should_split_gdn_mixed_prefill_decode(
                        enable_dp_attention=False,
                        **(base | override),
                    )
                )

    def apply_policy(
        self,
        runner,
        *,
        cuda=True,
        capability=(10, 0),
        cuda_version="13.0",
        flashinfer_available=True,
    ):
        with (
            patch.object(
                gdn_backend,
                "hybrid_gdn_config",
                return_value=runner.hybrid_gdn_config,
            ),
            patch.object(gdn_backend, "is_cuda", return_value=cuda),
            patch.object(torch.cuda, "get_device_capability", return_value=capability),
            patch.object(torch.version, "cuda", cuda_version),
            patch(
                "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer."
                "is_flashinfer_gdn_prefill_available",
                return_value=flashinfer_available,
            ),
        ):
            return flashinfer_gdn_prefill_default(runner)

    def test_selects_flashinfer_for_supported_sm100_gdn(self):
        self.assertEqual(self.apply_policy(make_runner()), "flashinfer")

    def test_selects_flashinfer_for_radix_cache_strategies(self):
        for strategy in ("no_buffer", "extra_buffer", "extra_buffer_lazy"):
            with self.subTest(strategy=strategy):
                runner = make_runner(
                    uses_mamba_radix_cache=True,
                    mamba_radix_cache_strategy=strategy,
                )
                self.assertEqual(self.apply_policy(runner), "flashinfer")

    def test_declines_when_the_prefill_backend_is_explicit(self):
        for backend in ("triton", "flashinfer", "cutedsl"):
            with self.subTest(backend=backend):
                runner = make_runner(linear_attn_prefill_backend=backend)
                self.assertIsNone(self.apply_policy(runner))

    def test_rejects_unsupported_capability(self):
        cases = (
            ("non_cuda", {}, {"cuda": False}),
            ("hopper", {}, {"capability": (9, 0)}),
            ("future_sm", {}, {"capability": (12, 0)}),
            ("cuda_12", {}, {"cuda_version": "12.9"}),
            ("fp32_state", {"state_dtype": torch.float32}, {}),
            ("key_dim", {"key_dim": 64}, {}),
            ("value_dim", {"value_dim": 64}, {}),
            ("missing_api", {}, {"flashinfer_available": False}),
        )
        for name, runner_args, hardware in cases:
            with self.subTest(name=name):
                self.assertIsNone(
                    self.apply_policy(make_runner(**runner_args), **hardware)
                )

    def test_rejects_gdn_config_without_qwen_head_dims(self):
        runner = make_runner()
        runner.hybrid_gdn_config = SimpleNamespace()
        self.assertIsNone(self.apply_policy(runner))

    def test_rejects_unvalidated_runtime_modes(self):
        cases = (
            ("non_triton_base", {"linear_attn_backend": "cutedsl"}),
            ("page_major_kv", {"enable_page_major_kv_layout": True}),
            ("dynamic_chunk", {"enable_dynamic_chunking": True}),
            ("unchunked", {"chunked_prefill_size": -1}),
            ("unknown_chunk", {"chunked_prefill_size": None}),
            ("large_chunk", {"chunked_prefill_size": 8193}),
        )
        for name, runner_args in cases:
            with self.subTest(name=name):
                self.assertIsNone(self.apply_policy(make_runner(**runner_args)))

    def test_builds_compact_checkpoint_plan_for_packed_sequences(self):
        forward_batch = SimpleNamespace(
            extend_seq_lens=torch.tensor([63, 64, 65, 127, 128, 129]),
            mamba_track_mask=torch.tensor([False, True, True, True, True, True]),
            # 65 on the 128-token sequence represents an interior S64
            # boundary encoded as S64 + 1 by the scheduler.
            mamba_track_seqlens=torch.tensor([63, 64, 65, 127, 65, 129]),
            extend_prefix_lens=torch.zeros(6, dtype=torch.int64),
        )
        metadata = SimpleNamespace(
            track_ssm_h_src=torch.empty(4),
            track_ssm_h_dst=torch.empty(4),
        )

        with patch(
            "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer."
            "get_server_args",
            return_value=SimpleNamespace(mamba_cache_chunk_size=64),
        ):
            maybe_build_flashinfer_checkpoint_plan(forward_batch, metadata, "cpu")

        torch.testing.assert_close(
            metadata.state_checkpoint_cu_starts,
            torch.tensor([0, 0, 1, 2, 3, 5, 7]),
        )
        torch.testing.assert_close(metadata.track_ssm_h_src, torch.tensor([1, 2, 3, 6]))
        self.assertEqual(metadata.num_state_checkpoints, 7)
        self.assertEqual(metadata.state_checkpoint_every_n_tokens, 64)

    def test_decode_tracking_without_h_source_skips_checkpoint_plan(self):
        backend = object.__new__(GDNAttnBackend)
        backend.device = "cpu"
        backend.kernel_dispatcher = SimpleNamespace(extend_uses_state_checkpoints=True)
        metadata = SimpleNamespace(has_mamba_track_mask=True, track_ssm_h_src=None)
        forward_batch = SimpleNamespace(
            mamba_track_mask=torch.tensor([True]),
            mamba_track_indices=torch.tensor([7]),
        )

        def init_base(instance, _forward_batch):
            instance.forward_metadata = metadata

        with patch.object(MambaAttnBackendBase, "init_forward_metadata", init_base):
            backend.init_forward_metadata(forward_batch)

        torch.testing.assert_close(metadata.conv_states_mask_indices, torch.tensor([7]))

    def test_tree_verify_uses_triton_kernel(self):
        flashinfer_kernel = MagicMock(supports_target_verify=True)
        with (
            patch.object(gdn_backend, "is_cuda", return_value=True),
            patch(
                "sglang.srt.layers.attention.linear.kernels.gdn_flashinfer."
                "FlashInferGDNKernel",
                return_value=flashinfer_kernel,
            ),
        ):
            dispatcher = GDNKernelDispatcher(
                LinearAttnKernelBackend.TRITON,
                LinearAttnKernelBackend.FLASHINFER,
            )

        self.assertIsInstance(dispatcher.tree_verify_kernel, TritonGDNKernel)

        tensor = sentinel.tensor
        with patch.object(
            dispatcher.tree_verify_kernel, "target_verify"
        ) as tree_verify:
            dispatcher.target_verify(
                *([tensor] * 7),
                ssm_states=tensor,
                cache_indices=tensor,
                query_start_loc=tensor,
                retrieve_parent_token=sentinel.parent_token,
            )

        tree_verify.assert_called_once()
        flashinfer_kernel.target_verify.assert_not_called()

    def test_mixed_extend_splits_prefill_and_decode_kernels(self):
        class FakeDispatcher:
            supports_packed_decode = False

            def __init__(self):
                self.extend_kwargs = None
                self.decode_kwargs = None

            def extend(self, **kwargs):
                self.extend_kwargs = kwargs
                tokens = kwargs["q"].shape[1]
                return torch.ones(1, tokens, 1, 2), None, None

            def decode(self, **kwargs):
                self.decode_kwargs = kwargs
                tokens = kwargs["q"].shape[1]
                return torch.full((1, tokens, 1, 2), 2.0)

        dispatcher = FakeDispatcher()
        backend = object.__new__(GDNAttnBackend)
        backend.kernel_dispatcher = dispatcher
        conv_states = torch.zeros(8, 6, 3)
        ssm_states = torch.zeros(8, 1, 2, 2)
        backend.req_to_token_pool = SimpleNamespace(
            mamba2_layer_cache=lambda _layer_id: SimpleNamespace(
                conv=[conv_states], temporal=ssm_states
            )
        )
        backend.forward_metadata = SimpleNamespace(
            query_start_loc=torch.tensor([0, 4, 5, 6], dtype=torch.int32),
            mamba_cache_indices=torch.tensor([5, 6, 7], dtype=torch.int32),
            num_mixed_prefill_reqs=1,
            num_mixed_prefill_tokens=4,
            num_mixed_decode_reqs=2,
            mixed_decode_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            has_mamba_track_mask=False,
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            retrieve_parent_token=None,
            state_checkpoint_cu_starts=None,
            num_state_checkpoints=0,
            state_checkpoint_every_n_tokens=0,
        )
        layer = SimpleNamespace(
            layer_id=0,
            conv_weights=torch.zeros(6, 4),
            bias=torch.zeros(6),
            activation="silu",
            q_dim=2,
            k_dim=2,
            v_dim=2,
            num_q_heads=1,
            num_k_heads=1,
            num_v_heads=1,
            head_q_dim=2,
            head_k_dim=2,
            head_v_dim=2,
            A_log=torch.zeros(1),
            dt_bias=torch.zeros(1),
        )
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_target_verify=lambda: False),
            extend_prefix_lens=torch.tensor([0, 9, 11]),
            extend_seq_lens_cpu=[4, 1, 1],
        )
        mixed_qkv = torch.arange(36, dtype=torch.float32).view(6, 6)
        a = torch.arange(6, dtype=torch.float32).view(6, 1)
        b = -a

        def fake_prefill_conv(x, *args, **kwargs):
            self.assertEqual(tuple(x.shape), (6, 4))
            torch.testing.assert_close(
                kwargs["query_start_loc"], torch.tensor([0, 4], dtype=torch.int32)
            )
            torch.testing.assert_close(
                kwargs["cache_indices"], torch.tensor([5], dtype=torch.int32)
            )
            self.assertEqual(kwargs["seq_lens_cpu"], [4])
            return x + 10

        def fake_decode_conv(x, *args, **kwargs):
            self.assertEqual(tuple(x.shape), (2, 6))
            torch.testing.assert_close(
                kwargs["conv_state_indices"],
                torch.tensor([6, 7], dtype=torch.int32),
            )
            return x + 20

        def fake_gating(_A_log, a_slice, b_slice, _dt_bias):
            self.assertEqual(tuple(a_slice.shape), (4, 1))
            self.assertEqual(tuple(b_slice.shape), (4, 1))
            return a_slice.unsqueeze(0), b_slice.unsqueeze(0)

        with (
            patch.object(gdn_backend, "is_cpu", return_value=True),
            patch.object(
                gdn_backend, "causal_conv1d_fn", side_effect=fake_prefill_conv
            ),
            patch.object(
                gdn_backend, "causal_conv1d_update", side_effect=fake_decode_conv
            ),
            patch.object(gdn_backend, "fused_gdn_gating", side_effect=fake_gating),
        ):
            output = backend.forward_extend(
                layer=layer,
                forward_batch=forward_batch,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
            )

        self.assertEqual(tuple(output.shape), (1, 6, 1, 2))
        torch.testing.assert_close(output[:, :4], torch.ones(1, 4, 1, 2))
        torch.testing.assert_close(output[:, 4:], torch.full((1, 2, 1, 2), 2.0))
        self.assertEqual(dispatcher.extend_kwargs["q"].shape[1], 4)
        self.assertEqual(dispatcher.decode_kwargs["q"].shape[1], 2)
        torch.testing.assert_close(
            dispatcher.decode_kwargs["query_start_loc"],
            torch.tensor([0, 1, 2], dtype=torch.int32),
        )
        torch.testing.assert_close(
            dispatcher.decode_kwargs["cache_indices"],
            torch.tensor([6, 7], dtype=torch.int32),
        )

    def test_mixed_all_decode_tbo_child_skips_prefill_kernels(self):
        class FakeDispatcher:
            supports_packed_decode = False

            def extend(self, **_kwargs):
                raise AssertionError("all-decode TBO child must not run extend")

            def decode(self, **kwargs):
                self.decode_kwargs = kwargs
                tokens = kwargs["q"].shape[1]
                return torch.full((1, tokens, 1, 2), 3.0)

        dispatcher = FakeDispatcher()
        backend = object.__new__(GDNAttnBackend)
        backend.kernel_dispatcher = dispatcher
        backend.forward_metadata = SimpleNamespace(
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            mamba_cache_indices=torch.tensor([6, 7], dtype=torch.int32),
            num_mixed_prefill_reqs=0,
            num_mixed_prefill_tokens=0,
            num_mixed_decode_reqs=2,
            mixed_decode_query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            state_checkpoint_cu_starts=None,
            num_state_checkpoints=0,
            state_checkpoint_every_n_tokens=0,
        )
        layer = SimpleNamespace(
            conv_weights=torch.zeros(6, 4),
            bias=torch.zeros(6),
            activation="silu",
            q_dim=2,
            k_dim=2,
            v_dim=2,
            num_q_heads=1,
            num_k_heads=1,
            num_v_heads=1,
            head_q_dim=2,
            head_k_dim=2,
            head_v_dim=2,
            A_log=torch.zeros(1),
            dt_bias=torch.zeros(1),
        )
        forward_batch = SimpleNamespace(
            extend_prefix_lens=torch.tensor([9, 11]),
            extend_seq_lens_cpu=[1, 1],
        )
        conv_states = torch.zeros(8, 6, 3)
        ssm_states = torch.zeros(8, 1, 2, 2)
        mixed_qkv = torch.arange(12, dtype=torch.float32).view(2, 6)

        with patch.object(
            gdn_backend,
            "causal_conv1d_update",
            side_effect=lambda x, *_args, **_kwargs: x,
        ):
            output = backend._forward_mixed_prefill_decode(
                layer=layer,
                forward_batch=forward_batch,
                mixed_qkv=mixed_qkv,
                a=torch.zeros(2, 1),
                b=torch.zeros(2, 1),
                conv_states=conv_states,
                ssm_states=ssm_states,
                conv_states_contig=conv_states,
                ssm_states_contig=ssm_states,
                state_cache_indices=torch.tensor([6, 7], dtype=torch.int32),
                needs_state_gather=False,
            )

        torch.testing.assert_close(output, torch.full((1, 2, 1, 2), 3.0))
        torch.testing.assert_close(
            dispatcher.decode_kwargs["cache_indices"],
            torch.tensor([6, 7], dtype=torch.int32),
        )


if __name__ == "__main__":
    unittest.main()
