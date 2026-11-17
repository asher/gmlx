"""The row of an image family's VAE, which the reader and the VAE modules
that keep their own row both read. Importing this module loads no mlx."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class VaeRow:
    """The VAE an image family decodes with."""
    title: str
    config_from_shapes: Callable    # torch shapes -> config, or None
    build: Callable                 # config -> nn.Module
    latent_channels: int
    latent_space: Callable | None = None    # family constants -> config
                                    # fields; None for a VAE outside the
                                    # image registry
    expected_unused: Callable | None = None     # config -> the tensors a
                                                # file may hold that stay
                                                # unread, name -> torch shape
    image_channels: int | None = None   # in and out, when the family fixes
                                        # them
    relative: Callable | None = None    # torch shapes -> what a file with
                                        # this tree holds instead, or None
    config_disagreements: Callable | None = None    # vae/config.json dict
                                        # -> fields that differ from the
                                        # VAE's constants
    fp32_keep: tuple = ()               # parameters kept in float32
    layout: Callable | None = None      # (name, torch shape) -> (axes or
                                        # None, module shape), for tensors
                                        # outside the image layout
    dtypes: tuple = ()                  # decode dtypes, the default first;
                                        # () for bfloat16 or float32
    own_names: bool = False             # read under the file's names, with
                                        # no rename to diffusers names
    refusal: Callable | None = None     # VaeHeader -> why gmlx does not run
                                        # the file, or None
