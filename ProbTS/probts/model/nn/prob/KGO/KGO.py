import torch
import torch.nn as nn
from einops import rearrange

from .xlstm.xlstm_block_stack import xLSTMBlockStack, xLSTMBlockStackConfig
from .xlstm.blocks.slstm.block import sLSTMBlockConfig
from .xlstm.blocks.slstm.layer import sLSTMLayerConfig
from .xlstm.components.feedforward import FeedForwardConfig
from .AUG import AUG
from .KoFM import KoFM
from .KoPE import KoPE
from .PatchEmbedding import PatchEmbedding
from .ResidualBlock import ResidualBlock
from .RevIN import RevIN
from .loss import crps_loss


class Model(nn.Module):
    """KGO (Sec. 3): deterministic and stochastic (KoFM) streams on the KoPE latents, fused by AUG.

    Shape legend used in the einops patterns below:
        B batch, V variates, L context length, H prediction length, P patch size,
        M forecast patches (H / P), D embedding dim, J samples per series (cs per sampling chunk).
    """

    def __init__(self, args):
        super().__init__()
        for name in ("context_length", "prediction_length"):
            assert getattr(args, name) % args.patch_size == 0, (
                f"{name} ({getattr(args, name)}) must be a multiple of input_patch_size ({args.patch_size})"
            )
        self.config = args
        self.M      = args.prediction_length // args.patch_size

        self.KoPE            = KoPE(args)
        self.patch_embedding = PatchEmbedding(args)
        self.revin           = RevIN(args.variates, affine=True)
        self.KoFM            = KoFM(args)
        self.AUG             = AUG(args)
        self.det_decoder     = ResidualBlock(
            in_dim     = args.embedding_dim,
            h_dim      = args.embedding_dim * args.hidden_mul,
            out_dim    = args.patch_size,
            num_layers = args.decoder_num_layers,
            dropout    = args.dropout,
        )

        slstm_config = sLSTMLayerConfig(
            num_heads          = args.num_heads,
            conv1d_kernel_size = args.conv1d_kernel_size,
            bias_init          = "powerlaw_blockdependent",
        )
        feedforward_config = FeedForwardConfig(
            proj_factor = args.ffn_proj_factor,
            act_fn      = "gelu",
        )
        block_config = sLSTMBlockConfig(
            slstm       = slstm_config,
            feedforward = feedforward_config,
        )
        encoder_config = xLSTMBlockStackConfig(
            slstm_block          = block_config,
            context_length       = args.context_length,
            num_blocks           = args.num_blocks,
            embedding_dim        = args.embedding_dim,
            add_post_blocks_norm = True,
            slstm_at             = "all",
        )
        self.seq_encoder = xLSTMBlockStack(encoder_config)

    @staticmethod
    def _mix(x_prob, x_det, gates):
        """Eq. (5): blend stochastic samples (B, J, H, V) with the deterministic forecast (B, H, V)
        using the AUG weights (B, 2, H, V)."""
        return x_prob * gates[:, 0:1] + x_det[:, None] * gates[:, 1:2]

    def _denorm(self, samples):
        """RevIN-denormalize samples (B, J, H, V)."""
        J = samples.shape[1]
        with self.revin.expanded(J):
            out = self.revin(samples.flatten(0, 1), mode="denorm")
        return out.unflatten(0, (-1, J))

    def _train_forward(self, c, x_det, gates, y):
        J, M = self.config.num_train_samples, self.M

        y_norm    = self.revin._normalize(y)
        y_patches = rearrange(y_norm, "B (M P) V -> (B M) V P", M=M)
        x_prob    = rearrange(self.KoFM(c, y_patches), "(B M J) V P -> B J (M P) V", M=M, J=J)

        samples = self._denorm(self._mix(x_prob, x_det, gates))
        return crps_loss(samples, y)

    def _inference_forward(self, c, x_det, gates, validation):
        B, H, V     = x_det.shape
        num_samples = self.config.num_train_samples if validation else self.config.num_sampling
        chunk       = min(self.config.num_samples_chunk or num_samples, num_samples)
        cond        = self.KoFM.condition(c)
        samples     = None

        # Sample in chunks to bound peak memory; the conditioning is shared by all chunks.
        for start in range(0, num_samples, chunk):
            cs     = min(chunk, num_samples - start)
            x_prob = rearrange(self.KoFM.sample(cond, cs), "(B M cs) V P -> B cs (M P) V", M=self.M, cs=cs)
            out    = self._denorm(self._mix(x_prob, x_det, gates))

            if samples is None:
                samples = torch.empty(B, num_samples, H, V, dtype=out.dtype, device=out.device)
            samples[:, start : start + cs].copy_(out)

        return samples

    def forward(self, x, y=None, validation=False):
        """x: (B, L, V). With y (B, H, V) returns the training loss,
        otherwise forecast samples (B, num_samples, H, V)."""
        B = x.shape[0]

        gates = self.AUG(x)
        x     = self.revin(x, mode="norm")

        z      = self.seq_encoder(self.patch_embedding(x))
        z_kope = self.KoPE(z)

        x_det = rearrange(self.det_decoder(z_kope), "(B V) M P -> B (M P) V", B=B)
        c     = rearrange(z_kope, "(B V) M D -> (B M) V D", B=B)

        if y is not None:
            return self._train_forward(c, x_det, gates, y)
        return self._inference_forward(c, x_det, gates, validation)
