"""Qwen3.5 family: attention, GDN, and verify paths."""

# The copied package's language module is an alias of the owned one, and
# the owned one imports the copied package's config. The copied package
# goes first so that its container never sees a half-built language module.
import gmlx._vendor.mlx_vlm.models.qwen3_5  # noqa: F401
