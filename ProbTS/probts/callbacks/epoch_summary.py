import os
import time

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback


_RESET = "\033[0m"
_BOLD_CYAN = "\033[1;36m"
_DIM = "\033[2m"
_BRIGHT = "\033[1m"


def _color_supported() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return True


class EpochSummaryCallback(Callback):
    """Print one curated, color-coded line at the end of each validation epoch."""

    def __init__(self, metrics=("train_loss", "val_CRPS", "val_ND"), use_color=None):
        self.metrics = tuple(metrics)
        self.use_color = _color_supported() if use_color is None else use_color
        self._epoch_start = None

    def _c(self, text: str, code: str) -> str:
        if not self.use_color:
            return text
        return f"{code}{text}{_RESET}"

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._epoch_start = time.time()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs or 0
        elapsed = time.time() - self._epoch_start if self._epoch_start is not None else 0.0

        tag = self._c(f"[Epoch {epoch:>2}/{max_epochs}]", _BOLD_CYAN)
        elapsed_str = self._c(f"{elapsed:.1f}s", _DIM)

        parts = [tag, elapsed_str]
        cm = trainer.callback_metrics
        for name in self.metrics:
            if name not in cm:
                continue
            val = cm[name]
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue
            label = self._c(f"{name}=", _DIM)
            value = self._c(f"{val:.4f}", _BRIGHT)
            parts.append(f"{label}{value}")

        print("  ".join(parts), flush=True)
