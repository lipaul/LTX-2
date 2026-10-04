"""Neighbourhood-attention transformer configuration for the diffusion video VAE.

Only the configuration enums live here.  The diffusion VAE implementation is
not part of this build: the decoder path in use is the convolutional VAE
(``ltx_core.model.video_vae.conv_video_decoder``), and the enum values below
are still accepted by the pipeline argument parsers.
"""

from ltx_core.model.video_vae.transformer.config import (
    DiffVAEBlockKind,
    DiffVAEConfig,
    DiffVAEMode,
    NAttentionKind,
)

__all__ = [
    "DiffVAEBlockKind",
    "DiffVAEConfig",
    "DiffVAEMode",
    "NAttentionKind",
]
