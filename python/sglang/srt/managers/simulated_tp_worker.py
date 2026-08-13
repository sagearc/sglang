# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0

"""Tensor-parallel worker for metadata-only simulated forward."""

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.simulated_model_runner import SimulatedModelRunner


class SimulatedTpModelWorker(TpModelWorker):
    def get_model_runner_class(self):
        return SimulatedModelRunner

    def forward_batch_generation(self, batch, *args, **kwargs):
        if batch is None:
            raise RuntimeError("Simulated forward requires the scheduled requests.")
        self.model_runner.prepare_simulated_tokens(batch.reqs)
        return super().forward_batch_generation(batch, *args, **kwargs)
