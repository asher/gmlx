"""Gemma 4 family: owned backbone and batched SDPA."""

# The copied package's language module is an alias of the owned one, and
# the owned one imports the copied package's config. The copied package
# goes first so that its container never sees a half-built language module.
import gmlx._vendor.mlx_vlm.models.gemma4  # noqa: F401
