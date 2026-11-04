import time
from typing import Any

import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks.callback import Callback


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class TimeCallback(Callback):
    """
        Trace the computation time.
    """
    def __init__(self):
        self.time_summary = {
            'train_batch_time': [],
            'val_batch_time': [],
            'test_batch_time': [],
            'test_forward_time': [],
        }
        self._original_forecast = None

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch begins."""
        _cuda_sync()
        self.train_start_time = time.time()

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch ends"""
        _cuda_sync()
        self.time_summary['train_batch_time'].append(time.time() - self.train_start_time)

    def on_validation_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the validation batch begins"""
        _cuda_sync()
        self.val_start_time = time.time()

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the validation batch ends"""
        _cuda_sync()
        self.time_summary['val_batch_time'].append(time.time() - self.val_start_time)

    def on_test_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch begins"""
        _cuda_sync()
        self.test_start_time = time.time()

    def on_test_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch ends"""
        _cuda_sync()
        self.time_summary['test_batch_time'].append(time.time() - self.test_start_time)

    # --- Forward-pass-only timing (isolates model inference from metrics) ---

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Warmup: run one untimed forward pass to trigger torch.matrix_exp
        # cache build, CUDA kernel JIT, and memory allocation before timing.
        try:
            batch = next(iter(trainer.test_dataloaders))
            batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)
            with torch.no_grad():
                pl_module.forecaster.forecast(batch)
            _cuda_sync()
        except Exception:
            pass  # if warmup fails, proceed without it

        original_forecast = pl_module.forecaster.forecast
        forward_times = self.time_summary['test_forward_time']

        def timed_forecast(*args, **kwargs):
            _cuda_sync()
            t0 = time.time()
            result = original_forecast(*args, **kwargs)
            _cuda_sync()
            forward_times.append(time.time() - t0)
            return result

        pl_module.forecaster.forecast = timed_forecast
        self._original_forecast = original_forecast

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._original_forecast is not None:
            pl_module.forecaster.forecast = self._original_forecast
            self._original_forecast = None
