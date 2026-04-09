"""Test that custom all-reduce coexists with NCCL symmetric memory.

Verifies that enabling both --enable-symm-mem and custom all-reduce (default)
does not crash and maintains accuracy. Previously, enabling symm-mem would
completely bypass custom AR, hurting decode throughput.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=600, suite="nightly-8-gpu-h200", nightly=True)

MODEL_PATH = "meta-llama/Llama-3.1-70B-Instruct"
SERVER_LAUNCH_TIMEOUT = 600


class TestSymmMemCustomARCoexistence(CustomTestCase):
    """Launch server with both symm-mem and custom AR enabled, verify accuracy."""

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL_PATH)
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tensor-parallel-size",
            "8",
            "--enable-symm-mem",
            # custom all-reduce is enabled by default (not passing --disable-custom-all-reduce)
            "--mem-fraction-static",
            "0.85",
            "--disable-radix-cache",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k_accuracy(self):
        """GSM8K accuracy should be >= 0.80 with both symm-mem and custom AR."""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=64,
        )
        metrics = run_eval(args)
        print(f"Eval accuracy of GSM8K (symm-mem + custom AR): {metrics=}")
        self.assertGreater(metrics["score"], 0.80)


if __name__ == "__main__":
    unittest.main()
