import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.simulated_model_runner import SimulatedModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams


class TestSimulatedModelRunner(unittest.TestCase):
    def _req(self, token_ids=..., *, output_ids=(), eos_token_ids=(99,)):
        custom_params = (
            {} if token_ids is ... else {"simulated_output_token_ids": token_ids}
        )
        return SimpleNamespace(
            output_ids=array("q", output_ids),
            eos_token_ids=set(eos_token_ids),
            sampling_params=SamplingParams(custom_params=custom_params),
        )

    def test_sample_returns_caller_tokens_then_eos_then_zero(self):
        runner = SimulatedModelRunner.__new__(SimulatedModelRunner)

        runner.prepare_simulated_tokens(
            [
                self._req([501, 502]),
                self._req([501, 502], output_ids=[501]),
                self._req([501, 502], output_ids=[501, 502]),
                self._req([501, 502], output_ids=[501, 502, 99]),
                self._req(),
            ]
        )
        sampled = runner.sample(
            None,
            SimpleNamespace(input_ids=torch.tensor([1], dtype=torch.long)),
        )

        self.assertEqual(sampled.tolist(), [501, 502, 99, 0, 99])


if __name__ == "__main__":
    unittest.main()
