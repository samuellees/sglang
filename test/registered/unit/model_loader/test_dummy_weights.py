import unittest

import torch

from sglang.srt.model_loader.weight_utils import initialize_dummy_weights


class _DummyMXFP8Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(8, 8, dtype=torch.float16))
        self.weight_scale_inv = torch.nn.Parameter(
            torch.empty(8, 2, dtype=torch.uint8), requires_grad=False
        )
        self.weight_scale_inv.format_ue8m0 = True
        self.register_buffer("unrelated_integer", torch.full((4,), 23, dtype=torch.uint8))


class TestInitializeDummyWeights(unittest.TestCase):
    def test_initializes_ue8m0_scale_without_touching_other_integers(self):
        model = _DummyMXFP8Module()

        initialize_dummy_weights(model)

        self.assertTrue(torch.all(model.weight_scale_inv == 127))
        self.assertTrue(torch.all(model.unrelated_integer == 23))
        self.assertTrue(torch.isfinite(model.weight).all())


if __name__ == "__main__":
    unittest.main()
