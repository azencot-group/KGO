from types import SimpleNamespace

import torch
from torch import optim
from transformers import get_cosine_schedule_with_warmup

from probts.model.forecaster.forecaster import Forecaster
from probts.model.nn.prob.KGO.KGO import Model as KGO


class KGOModel(Forecaster):
    def __init__(
        self,
        dropout,
        num_blocks,
        input_patch_size,
        embedding_dim,
        kofm_dim,
        num_heads,
        num_train_samples,
        context_length,
        prediction_length,
        num_sampling,
        conv1d_kernel_size,
        ffn_proj_factor,
        decoder_num_layers,
        hidden_mul,
        num_samples_chunk=None,
        **kwargs,
    ):
        super().__init__(
            context_length=context_length, prediction_length=prediction_length, **kwargs
        )
        config = SimpleNamespace(
            **kwargs,
            num_samples_chunk=num_samples_chunk,
            dropout=dropout,
            num_blocks=num_blocks,
            patch_size=input_patch_size,
            embedding_dim=embedding_dim,
            kofm_dim=kofm_dim,
            num_heads=num_heads,
            num_train_samples=num_train_samples,
            context_length=context_length,
            prediction_length=prediction_length,
            num_sampling=num_sampling,
            conv1d_kernel_size=conv1d_kernel_size,
            ffn_proj_factor=ffn_proj_factor,
            decoder_num_layers=decoder_num_layers,
            hidden_mul=hidden_mul,
            variates=self.target_dim,
        )
        self.model = KGO(config)

    def configure_optimizers(self, lr, total_steps):
        generator_params = list(self.model.KoFM.transport.parameters())
        generator_ids = {id(p) for p in generator_params}
        other_params = [p for p in self.parameters() if id(p) not in generator_ids]
        optimizer = optim.AdamW(
            [
                {"params": other_params, "lr": lr},
                {"params": generator_params, "lr": lr * 0.1},
            ],
            betas=(0.9, 0.95),
            eps=1e-10,
            weight_decay=0.1,
        )
        warmup_steps = int(0.10 * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,
        )
        return optimizer, scheduler

    def loss(self, batch_data):
        self.model.train()
        x = batch_data.past_target_cdf[:, -self.context_length :, :]
        target = batch_data.future_target_cdf
        loss = self.model(x, target)
        return loss

    def forecast(self, batch_data, num_samples=None, validation=False):
        x = batch_data.past_target_cdf[:, -self.context_length :, :]
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x, validation=validation)
        return outputs
