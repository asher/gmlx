# SPDX-License-Identifier: Apache-2.0 AND MIT
# SPDX-FileCopyrightText: Copyright 2026 OpenMOSS Team
# The constants of vae/config.json in OpenMOSS-Team/MOSS-Image-SFT at
# revision fc5de0d5 (Apache-2.0; see licenses/MOSS-Image-LICENSE).
"""The MOSS-Image autoencoder: the FLUX.2 autoencoder with a narrower
decoder.

The encoder, the quant convs, the BatchNorm statistics and the 2x2
packing of model space are the FLUX.2 autoencoder's. The decoder runs at
[96, 192, 384, 384] channels per level from the full resolution side,
where the encoder runs at [128, 256, 512, 512], so the module is
``flux2_ae.Flux2AE`` at those decoder widths.
"""

from __future__ import annotations

from . import flux2_ae

# Decoder widths from the official vae/config.json, where the other config
# values equal the FLUX.2 autoencoder's constants.
DECODER = (96, 192, 384, 384)
CONSTANTS = {**flux2_ae.CONSTANTS, "decoder_block_out_channels": DECODER}
CONSTANTS_SOURCE = ("OpenMOSS-Team/MOSS-Image-SFT",
                    "fc5de0d56fc910cbca6f6625a70f5f582b26deb1")


def config_from_shapes(shapes: dict) -> flux2_ae.Flux2AEConfig | None:
    """The config from diffusers-named torch shapes, or None when the names
    are not this autoencoder's: the FLUX.2 tree with the decoder widths of
    ``DECODER``."""
    cfg = flux2_ae.tree_config(shapes)
    if cfg is None or cfg.decoder_block_out_channels != DECODER:
        return None
    return cfg


def relative(shapes: dict) -> str | None:
    """What a FLUX.2 tree holds whose decoder is neither as wide as its
    encoder nor at ``DECODER``, or None."""
    cfg = flux2_ae.tree_config(shapes)
    if cfg is None or cfg.decoder_block_out_channels in ((), DECODER):
        return None
    return (f"a FLUX.2 autoencoder with decoder widths "
            f"{list(cfg.decoder_block_out_channels)}")


def config_disagreements(config: dict) -> list:
    """The fields of a vae/config.json that differ from ``CONSTANTS``. A
    field the config leaves out is none."""
    return flux2_ae.constant_disagreements(config, CONSTANTS)
