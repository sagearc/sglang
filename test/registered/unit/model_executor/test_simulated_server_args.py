import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.server_args import ServerArgs


class TestSimulatedServerArgs(unittest.TestCase):
    def _args(self, **overrides):
        args = ServerArgs(model_path="dummy", simulate_forward=True)
        for name, value in overrides.items():
            setattr(args, name, value)
        return args

    def _handle(self, args):
        model_config = SimpleNamespace(
            is_multimodal=False,
            is_generation=True,
            attention_arch=AttentionArch.MHA,
            hf_config=SimpleNamespace(architectures=["Qwen3ForCausalLM"]),
        )
        with (
            patch("sglang.srt.server_args.current_platform.is_cpu", return_value=True),
            patch.object(args, "get_model_config", return_value=model_config),
            patch(
                "sglang.srt.configs.model_config.is_minimax_sparse",
                return_value=False,
            ),
            patch(
                "sglang.srt.server_args.get_linear_attn_spec_by_arch",
                return_value=None,
            ),
        ):
            args._handle_simulate_forward()

    def test_sets_metadata_only_execution_defaults(self):
        args = self._args()

        self._handle(args)

        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.load_format, "dummy")
        self.assertEqual(args.kv_cache_dtype, "auto")
        self.assertTrue(args.disable_overlap_schedule)
        self.assertTrue(args.disable_cuda_graph)
        self.assertTrue(args.skip_server_warmup)

    def test_rejects_features_that_require_model_execution(self):
        cases = {
            "data parallel": {"dp_size": 2},
            "expert parallel": {"ep_size": 2},
            "lora": {"enable_lora": True},
            "disaggregation": {"disaggregation_mode": "prefill"},
            "hierarchical cache": {"enable_hierarchical_cache": True},
            "hidden states": {"enable_return_hidden_states": True},
            "routed experts": {"enable_return_routed_experts": True},
        }

        for name, overrides in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                self._handle(self._args(**overrides))


if __name__ == "__main__":
    unittest.main()
