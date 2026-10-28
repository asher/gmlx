"""Video diffusion models from GGUF files.

Nothing under this package is imported by ``import gmlx``. Families live in
``gmlx.video.families``, one directory each, found by ``registry.families``.
The package reuses the image stack's loader, ops and pacing, and leaves the
image 4-D contracts unchanged.
"""
