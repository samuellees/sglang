import unittest

import torch

from sglang.srt.layers.attention.trtllm_mha_backend import (
    TRTLLMMHAMixedBatchLayout,
    resolve_trtllm_mha_mixed_batch_layout,
    should_use_context_for_mixed_decode,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


def _make_forward_batch(
    extend_seq_lens: list[int], decode_batch_size: int
) -> ForwardBatch:
    batch_size = len(extend_seq_lens)
    num_tokens = sum(extend_seq_lens)
    return ForwardBatch(
        forward_mode=ForwardMode.MIXED,
        batch_size=batch_size,
        input_ids=torch.arange(num_tokens),
        req_pool_indices=torch.arange(batch_size),
        seq_lens=torch.tensor(extend_seq_lens),
        out_cache_loc=torch.arange(num_tokens),
        seq_lens_sum=num_tokens,
        extend_seq_lens_cpu=extend_seq_lens,
        mixed_decode_batch_size=decode_batch_size,
    )


class TestTRTLLMMHAMixedBatchLayout(unittest.TestCase):
    def test_high_local_gqa_uses_full_context_fallback(self):
        self.assertFalse(should_use_context_for_mixed_decode(4, 1))
        self.assertTrue(should_use_context_for_mixed_decode(8, 1))
        with self.assertRaisesRegex(ValueError, "num_kv_heads must be positive"):
            should_use_context_for_mixed_decode(4, 0)

    def test_prefill_decode_split(self):
        batch = _make_forward_batch([3, 5, 1, 1, 1], decode_batch_size=3)

        self.assertEqual(
            resolve_trtllm_mha_mixed_batch_layout(batch),
            TRTLLMMHAMixedBatchLayout(
                prefill_batch_size=2,
                prefill_num_tokens=8,
                prefill_max_seq_len_q=5,
                decode_batch_size=3,
            ),
        )

    def test_rejects_non_single_token_decode_tail(self):
        batch = _make_forward_batch([3, 1, 2], decode_batch_size=2)

        with self.assertRaisesRegex(ValueError, "one token per appended decode"):
            resolve_trtllm_mha_mixed_batch_layout(batch)

    def test_all_decode_tbo_child(self):
        batch = _make_forward_batch([1, 1], decode_batch_size=2)

        self.assertEqual(
            resolve_trtllm_mha_mixed_batch_layout(batch),
            TRTLLMMHAMixedBatchLayout(
                prefill_batch_size=0,
                prefill_num_tokens=0,
                prefill_max_seq_len_q=1,
                decode_batch_size=2,
            ),
        )


if __name__ == "__main__":
    unittest.main()
