# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0

"""Metadata-only model runner for scheduler and KV-cache simulation."""

from __future__ import annotations

import logging
from typing import Iterable, Optional

import torch

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.model_runner import ModelRunner, ModelRunnerOutput
from sglang.srt.model_executor.model_runner_components.layer_setup import (
    adjust_hybrid_swa_layer_ids,
    resolve_layer_indices,
)
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    resolve_sliding_window_size,
)
from sglang.srt.model_loader.loader import _initialize_model
from sglang.srt.model_loader.utils import set_default_torch_dtype

logger = logging.getLogger(__name__)


class SimulatedWeightManager:
    def __getattr__(self, name: str):
        raise RuntimeError(
            f"Weight operation {name!r} is unavailable with --simulate-forward."
        )


class SimulatedModelRunner(ModelRunner):
    """Runs SGLang scheduling and cache accounting without model computation."""

    def init_threads_binding(self):
        # No model kernels execute, so CPU affinity and compiled CPU kernels are
        # unnecessary. This also keeps the metadata-only mode portable.
        self.local_omp_cpuid = None

    def initialize(self):
        self.pending_token_ids = None
        self.init_memory_saver_adapter()
        load_config = LoadConfig(load_format=LoadFormat.DUMMY)
        with set_default_torch_dtype(self.model_config.dtype), torch.device("meta"):
            self.model = _initialize_model(self.model_config, load_config)
        non_meta_parameter = next(
            (
                (name, parameter.device)
                for name, parameter in self.model.named_parameters()
                if parameter.device.type != "meta"
            ),
            None,
        )
        if non_meta_parameter is not None:
            name, device = non_meta_parameter
            raise RuntimeError(
                f"Metadata-only model parameter {name!r} was created on {device}, "
                "not meta."
            )
        self.model.eval()
        logger.info(
            "Initialized metadata-only %s on meta with %d parameters.",
            type(self.model).__name__,
            sum(parameter.numel() for parameter in self.model.parameters()),
        )
        self.layer_info = resolve_layer_indices(
            model=self.model,
            model_config=self.model_config,
            is_draft_worker=self.is_draft_worker,
            spec_algorithm=self.spec_algorithm,
        )
        adjust_hybrid_swa_layer_ids(
            model_config=self.model_config,
            start_layer=self.layer_info.start_layer,
            end_layer=self.layer_info.end_layer,
            is_hybrid_swa=self.is_hybrid_swa,
        )
        self.sliding_window_size = resolve_sliding_window_size(
            self.model, self.model_config
        )
        self.prefill_aware_swa = (
            hasattr(self.model, "is_prefill_aware_swa")
            and self.model.is_prefill_aware_swa()
        )
        self.dtype = self.model_config.dtype
        self.weight_load_mem_usage = 0.0
        self.lora_manager = None
        self.eplb_manager = None
        self.expert_backup_client = None
        self.hisparse_coordinator = None
        self.configure_kv_cache_dtype()

    def check_quantized_moe_compatibility(self):
        return

    def init_weight_updater(self):
        self.weight_updater = SimulatedWeightManager()

    def init_weight_exporter(self):
        self.weight_exporter = SimulatedWeightManager()

    def _init_post_memory_pool_components(self):
        self.canary_manager = None
        self.init_ngram_embedding_manager()
        self.hisparse_coordinator = None
        self.graph_shared_output = None

    def init_attention_backends(self):
        self.attn_backend = None
        self.decode_attn_backend = None
        self.decode_attn_backend_group = None
        # ForwardBatch uses this label to build token positions.
        self.prefill_attention_backend_str = "torch_native"
        self.decode_attention_backend_str = "torch_native"

    def init_cuda_graphs(self, capture_decode_cuda_graph: bool = True):
        del capture_decode_cuda_graph
        self.eager_runner = None
        self.prefill_cuda_graph_runner = None
        self.decode_cuda_graph_runner = None
        self.graph_memory_usage = {}
        self.graph_time_usage = {}

    def forward(
        self,
        forward_batch,
        skip_attn_backend_init: Optional[bool] = None,
        pp_proxy_tensors=None,
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        del (
            skip_attn_backend_init,
            reinit_attn_backend,
            split_forward_count,
        )
        if pp_proxy_tensors is not None:
            raise RuntimeError(
                "--simulate-forward does not support pipeline parallelism."
            )
        if forward_batch.return_logprob:
            raise ValueError("--simulate-forward does not support logprobs.")
        self.forward_pass_id += 1
        return ModelRunnerOutput(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            can_run_graph=False,
        )

    def prepare_simulated_tokens(self, reqs: Iterable):
        """Select this step's caller-provided token for each scheduled request."""
        next_token_ids = []
        for req in reqs:
            params = req.sampling_params.custom_params or {}
            token_ids = params.get("simulated_output_token_ids")
            if not isinstance(token_ids, list) or any(
                type(token_id) is not int for token_id in token_ids
            ):
                token_ids = []
            output_index = len(req.output_ids)
            if output_index < len(token_ids):
                next_token_id = token_ids[output_index]
            elif output_index == len(token_ids) and req.eos_token_ids:
                next_token_id = next(iter(req.eos_token_ids))
            else:
                next_token_id = 0
            next_token_ids.append(next_token_id)
        self.pending_token_ids = next_token_ids

    def sample(self, logits_output, forward_batch) -> torch.Tensor:
        del logits_output
        next_token_ids = self.pending_token_ids
        if next_token_ids is None:
            raise RuntimeError("Simulated token IDs were not prepared for this batch.")
        self.pending_token_ids = None

        return torch.tensor(
            next_token_ids,
            dtype=torch.long,
            device=forward_batch.input_ids.device,
        )
