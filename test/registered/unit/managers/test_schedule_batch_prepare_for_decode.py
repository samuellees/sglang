import types
import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.managers.overlap_utils import FutureMap  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req():
    return types.SimpleNamespace(
        decode_batch_idx=0,
        kv_committed_len=3,
        kv=types.SimpleNamespace(kv_allocated_len=8),
    )


def _make_decode_batch():
    batch = ScheduleBatch(reqs=[_make_req(), _make_req()])
    batch.device = "cpu"
    batch.model_config = types.SimpleNamespace(is_encoder_decoder=False)
    batch.enable_overlap = False
    batch.spec_algorithm = types.SimpleNamespace(is_none=lambda: True)
    batch.sampling_info = types.SimpleNamespace(
        penalizer_orchestrator=types.SimpleNamespace(is_required=False)
    )
    batch.hisparse_coordinator = None
    batch.seq_lens = torch.tensor([3, 5], dtype=torch.int64)
    batch.seq_lens_cpu = torch.tensor([3, 5], dtype=torch.int64)
    batch.orig_seq_lens = torch.tensor([3, 5], dtype=torch.int32)
    return batch


def _make_eagle_decode_batch():
    batch = _make_decode_batch()
    batch.spec_algorithm = types.SimpleNamespace(
        is_none=lambda: False,
        is_eagle=lambda: True,
        is_frozen_kv_mtp=lambda: False,
    )
    return batch


class TestPrepareForDecodeSeqLensOwnership(unittest.TestCase):
    def test_decode_seq_lens_bump_is_out_of_place(self):
        """Each prepare_for_decode call rebinds seq-lens tensors to new +1 objects without mutating the old ones."""
        batch = _make_decode_batch()

        server_args = types.SimpleNamespace(
            enable_mamba_extra_buffer=lambda: False,
        )
        with (
            patch(
                "sglang.srt.managers.schedule_batch.alloc_for_decode",
                return_value=torch.tensor([6, 7], dtype=torch.int64),
            ),
            patch(
                "sglang.srt.managers.schedule_batch.get_server_args",
                return_value=server_args,
            ),
        ):
            for step in range(1, 3):
                prev_seq_lens = batch.seq_lens
                prev_seq_lens_cpu = batch.seq_lens_cpu
                prev_orig_seq_lens = batch.orig_seq_lens
                prev_values = (
                    prev_seq_lens.clone(),
                    prev_seq_lens_cpu.clone(),
                    prev_orig_seq_lens.clone(),
                )

                batch.prepare_for_decode()

                self.assertIsNot(batch.seq_lens, prev_seq_lens)
                self.assertIsNot(batch.seq_lens_cpu, prev_seq_lens_cpu)
                self.assertIsNot(batch.orig_seq_lens, prev_orig_seq_lens)
                expected = torch.tensor([3 + step, 5 + step], dtype=torch.int64)
                self.assertTrue(torch.equal(batch.seq_lens, expected))
                self.assertTrue(torch.equal(batch.seq_lens_cpu, expected))
                self.assertTrue(
                    torch.equal(batch.orig_seq_lens, expected.to(torch.int32))
                )
                self.assertTrue(torch.equal(prev_seq_lens, prev_values[0]))
                self.assertTrue(torch.equal(prev_seq_lens_cpu, prev_values[1]))
                self.assertTrue(torch.equal(prev_orig_seq_lens, prev_values[2]))

    def test_mixed_eagle_resident_rows_reuse_speculative_kv_reserve(self):
        """Mixed EAGLE executes through target prefill, so its concrete target slot must come from the speculative KV reserve."""
        batch = _make_eagle_decode_batch()
        batch.req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
        req_to_token = torch.zeros((2, 8), dtype=torch.int64)
        req_to_token[0, 3] = 6
        req_to_token[1, 5] = 7
        batch.req_to_token_pool = types.SimpleNamespace(req_to_token=req_to_token)
        server_args = types.SimpleNamespace(
            enable_mamba_extra_buffer=lambda: False,
        )
        out_cache_loc = torch.tensor([6, 7], dtype=torch.int64)

        def prepare_spec_reserve(prepared_batch):
            for req in prepared_batch.reqs:
                req.decode_batch_idx += 1

        with (
            patch(
                "sglang.srt.managers.schedule_batch.alloc_for_decode",
                return_value=out_cache_loc,
            ) as alloc_for_decode,
            patch(
                "sglang.srt.managers.schedule_batch.get_server_args",
                return_value=server_args,
            ),
            patch(
                "sglang.srt.speculative.spec_utils.spec_prepare_for_decode",
                side_effect=prepare_spec_reserve,
            ) as spec_prepare_for_decode,
        ):
            batch.prepare_for_decode(for_mixed_chunk=True)

        spec_prepare_for_decode.assert_called_once_with(batch)
        alloc_for_decode.assert_not_called()
        self.assertTrue(torch.equal(batch.out_cache_loc, out_cache_loc))
        self.assertTrue(
            torch.equal(batch.seq_lens, torch.tensor([4, 6], dtype=torch.int64))
        )
        self.assertEqual([req.kv_committed_len for req in batch.reqs], [4, 4])
        self.assertEqual([req.decode_batch_idx for req in batch.reqs], [1, 1])
        self.assertEqual([req.kv.kv_allocated_len for req in batch.reqs], [8, 8])


class TestMixedSpecSeqLensResolution(unittest.TestCase):
    def test_force_cpu_materializes_gpu_only_published_lengths(self):
        future_map = object.__new__(FutureMap)
        future_map.new_seq_lens_buf = torch.tensor([11, 13, 17], dtype=torch.int64)
        future_map.needs_cpu_seq_lens = False
        future_map.publish_ready = None
        future_map.fwd_prepare_d2h_stream = None

        batch = types.SimpleNamespace(
            spec_info=types.SimpleNamespace(
                future_indices=torch.tensor([2, 0], dtype=torch.int64)
            ),
            seq_lens=None,
            seq_lens_cpu=None,
            seq_lens_sum=None,
        )

        future_map.resolve_seq_lens_cpu(batch, force_cpu=True)

        expected = torch.tensor([17, 11], dtype=torch.int64)
        self.assertTrue(torch.equal(batch.seq_lens, expected))
        self.assertTrue(torch.equal(batch.seq_lens_cpu, expected))
        self.assertEqual(batch.seq_lens_sum, 28)


if __name__ == "__main__":
    unittest.main()
