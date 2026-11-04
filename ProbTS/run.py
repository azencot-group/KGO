import os
import warnings
import torch
import logging
from probts.data import ProbTSDataModule
from probts.model.forecast_module import ProbTSForecastModule
from probts.callbacks import MemoryCallback, TimeCallback, EpochSummaryCallback
from probts.utils import find_best_epoch
from lightning.pytorch import seed_everything
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from probts.utils.save_utils import save_exp_summary, save_csv

MULTI_HOR_MODEL = ["ElasTST", "Autoformer"]

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision("high")

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ProbTSCli(LightningCLI):
    def add_arguments_to_parser(self, parser):
        data_to_model_link_args = [
            "scaler",
            "train_pred_len_list",
        ]
        data_to_forecaster_link_args = [
            "target_dim",
            "history_length",
            "context_length",
            "prediction_length",
            "train_pred_len_list",
            "lags_list",
            "freq",
            "time_feat_dim",
            "global_mean",
            "dataset",
        ]
        for arg in data_to_model_link_args:
            parser.link_arguments(
                f"data.data_manager.{arg}", f"model.{arg}", apply_on="instantiate"
            )
        for arg in data_to_forecaster_link_args:
            parser.link_arguments(
                f"data.data_manager.{arg}",
                f"model.forecaster.init_args.{arg}",
                apply_on="instantiate",
            )

    def init_exp(self):
        config_args = self.config
        self.model_args = str(vars(self.model.forecaster.model.config))
        self.seed = config_args.seed_everything

        task = os.environ.get(
            "PROBTS_TASK",
            "stsf" if self.datamodule.data_manager.context_length <= 30 else "ltsf",
        )

        # Optional tag for independently named runs.
        custom_tag = os.environ.get("PROBTS_TAG")
        if custom_tag:
            self.tag = custom_tag
        elif self.datamodule.data_manager.multi_hor:
            assert self.model.forecaster.name in MULTI_HOR_MODEL, (
                f"Only support multi-horizon setting for {MULTI_HOR_MODEL}"
            )

            ctx_str = "-".join(
                [str(i) for i in self.datamodule.data_manager.train_ctx_len_list]
            )
            pred_str = "-".join(
                [str(i) for i in self.datamodule.data_manager.train_pred_len_list]
            )
            self.tag = "_".join(
                [
                    task,
                    self.datamodule.data_manager.dataset,
                    self.model.forecaster.name,
                    ctx_str,
                    pred_str,
                    str(config_args.seed_everything),
                ]
            )
        else:
            self.tag = "_".join(
                [
                    task,
                    self.datamodule.data_manager.dataset,
                    self.model.forecaster.name,
                    str(self.datamodule.data_manager.context_length),
                    str(self.datamodule.data_manager.prediction_length),
                    str(config_args.seed_everything),
                ]
            )

        log.info(f"Root dir is {self.trainer.default_root_dir}, exp tag is {self.tag}")

        if not os.path.exists(self.trainer.default_root_dir):
            os.makedirs(self.trainer.default_root_dir)

        self.save_dict = f"{self.trainer.default_root_dir}/{self.tag}"
        os.makedirs(self.save_dict, exist_ok=True)
        # Lightning exposes this directory as a read-only property. Isolate its
        # SaveConfigCallback as well as logs/checkpoints for concurrent runs.
        self.trainer._default_root_dir = self.save_dict

        if self.model.load_from_ckpt is not None:
            # if the checkpoint file is not assigned, find the best epoch in the current folder
            if ".ckpt" not in self.model.load_from_ckpt:
                sidecar = os.path.join(self.model.load_from_ckpt, "best_ckpt.txt")
                if os.path.exists(sidecar):
                    with open(sidecar, "r") as f:
                        best_path = f.read().strip()
                    # Accept either a bare filename or a full path
                    if not os.path.isabs(best_path):
                        best_path = os.path.join(self.model.load_from_ckpt, best_path)
                    print("using best ckpt from sidecar", best_path)
                    self.model.load_from_ckpt = best_path
                else:
                    _, best_ckpt = find_best_epoch(self.model.load_from_ckpt)
                    print("find best ckpt ", best_ckpt)
                    self.model.load_from_ckpt = os.path.join(
                        self.model.load_from_ckpt, best_ckpt
                    )

            log.info(f"Loading pre-trained checkpoint from {self.model.load_from_ckpt}")
            # Get the forecaster from the current model before loading the checkpoint
            current_forecaster = self.model.forecaster
            self.model = ProbTSForecastModule.load_from_checkpoint(
                self.model.load_from_ckpt,
                learning_rate=config_args.model.learning_rate,
                scaler=self.datamodule.data_manager.scaler,
                forecaster=current_forecaster,
                no_training=self.model.forecaster.no_training,
                sampling_weight_scheme=self.model.sampling_weight_scheme,
            )

        # Set callbacks
        self.memory_callback = MemoryCallback()
        self.time_callback = TimeCallback()

        callbacks = [
            self.memory_callback,
            self.time_callback,
            EpochSummaryCallback(),
        ]

        if not self.model.forecaster.no_training:
            if self.datamodule.dataset_val is None:  # if the validation set is empty
                monitor = "train_loss"
            else:
                # not using reweighting scheme for loss
                if self.model.sampling_weight_scheme in ["none", "fix"]:
                    monitor = "val_CRPS"
                else:
                    monitor = "val_weighted_ND"

            # Set callbacks
            self.checkpoint_callback = ModelCheckpoint(
                dirpath=f"{self.save_dict}/ckpt",
                filename="{epoch}-{val_CRPS:.6f}",
                every_n_epochs=1,
                monitor=monitor,
                save_top_k=1,
                save_last=True,
                enable_version_counter=False,
            )

            callbacks.append(self.checkpoint_callback)

        self.set_callbacks(callbacks)

        # EpochSummaryCallback supersedes Lightning's tqdm bar; drop the bar so
        # we don't get the wall-of-metrics line on top of our clean summary.
        self.trainer.callbacks[:] = [
            c
            for c in self.trainer.callbacks
            if "ProgressBar" not in c.__class__.__name__
        ]

    def set_callbacks(self, callbacks):
        # Replace built-in callbacks with custom callbacks
        custom_callbacks_name = [c.__class__.__name__ for c in callbacks]
        for c in self.trainer.callbacks:
            if c.__class__.__name__ in custom_callbacks_name:
                self.trainer.callbacks.remove(c)
        for c in callbacks:
            self.trainer.callbacks.append(c)
        for c in self.trainer.callbacks:
            if c.__class__.__name__ == "ModelSummary":
                self.model_summary_callback = c

    def set_fit_mode(self):
        self.trainer.logger = CSVLogger(
            save_dir=f"{self.save_dict}/logs",
            name=self.tag,
            version="fit",
        )

    def set_test_mode(self):
        self.trainer.logger = CSVLogger(
            save_dir=f"{self.save_dict}/logs",
            name=self.tag,
            version="test",
        )

        if not self.model.forecaster.no_training:
            self.ckpt = self.checkpoint_callback.best_model_path
            log.info(f"Loading best checkpoint from {self.ckpt}")
            # Persist the Lightning-selected best path so inference-only runs
            # can load exactly the same checkpoint regardless of which metric
            # was monitored (filename only embeds val_CRPS).
            try:
                sidecar = os.path.join(f"{self.save_dict}/ckpt", "best_ckpt.txt")
                with open(sidecar, "w") as f:
                    f.write(self.ckpt)
            except Exception as e:
                log.warning(f"Could not write best_ckpt.txt sidecar: {e}")
            # Get the forecaster from the trained model before loading the checkpoint
            trained_forecaster = self.model.forecaster
            trained_sampling_scheme = self.model.sampling_weight_scheme
            self.model = ProbTSForecastModule.load_from_checkpoint(
                self.ckpt,
                scaler=self.datamodule.data_manager.scaler,
                forecaster=trained_forecaster,
                no_training=trained_forecaster.no_training,
                sampling_weight_scheme=trained_sampling_scheme,
            )

    def run(self):
        self.init_exp()

        if not self.model.forecaster.no_training:
            self.set_fit_mode()
            if self.datamodule.dataset_val is None:  # if the validation set is empty
                self.trainer.fit(
                    model=self.model,
                    train_dataloaders=self.datamodule.train_dataloader(),
                )
            else:
                self.trainer.fit(model=self.model, datamodule=self.datamodule)

            inference = False
        else:
            inference = True

        self.set_test_mode()
        # Reseed at the start of the test stage so the RNG state entering
        # test is a function of only the seed, not of whether training ran.
        seed_everything(self.seed, workers=True)
        self.trainer.test(model=self.model, datamodule=self.datamodule)

        save_exp_summary(self, inference=inference)

        ctx_len = self.datamodule.data_manager.context_length
        if self.datamodule.data_manager.multi_hor:
            ctx_len = ctx_len[0]

        save_csv(self.model_args, self.save_dict, self.model, ctx_len)


def main():
    cli = ProbTSCli(
        datamodule_class=ProbTSDataModule,
        model_class=ProbTSForecastModule,
        save_config_kwargs={"overwrite": True},
        run=False,
    )
    cli.run()


if __name__ == "__main__":
    main()
