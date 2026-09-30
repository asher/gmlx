"""Vendored third-party packages (see THIRD_PARTY_NOTICES.md).

mlx_vlm and mlx_lm: mlx-vlm 0.7.3 and mlx-lm 0.31.3, copied whole by
scripts/vendor_upstream.py from the installed distributions, each file
checked against the sha256 its dist-info RECORD gives, and each package
with its LICENSE. Every file is upstream's, with only its module
references rewritten from ``mlx_vlm`` and ``mlx_lm`` to
``gmlx._vendor.mlx_vlm`` and ``gmlx._vendor.mlx_lm``. From then on the copy
is gmlx code: a fix goes in place, marked with a ``# gmlx:`` comment and
covered by a test. manifest.json records the sha256 upstream shipped for
each file and the file's status, and ``vendor_upstream.py verify`` checks
the tree against it. A later upstream release reaches the copy only
through a decision to port each change, with its test.

misaki: Kokoro's G2P, snapshotted at hexgrad/misaki@fba1236 (0.9.4 + the
unreleased Python-3.13 support). PyPI's misaki caps at <3.13, so the snapshot
carries the version support and two local fixes, each marked `gmlx:` in
misaki/en.py:

* torch/transformers back ``FallbackNetwork`` alone - a transformer G2P mode
  Kokoro never enables, since it always passes its own espeak fallback - so
  they load on first use instead of at import. At module scope they charged
  every ``gmlx[tts]`` install ~2 GB of torch that nothing then ran, and (as
  torch is not a declared dependency of the extra) the import failed outright
  unless ``gmlx[stt]`` happened to pull it in.
* the spacy pipeline resolves through the Hugging Face mirror rather than
  ``spacy.cli.download``, which shells out to pip - absent from ``uv tool``
  and pipx environments, where its response is a bare ``sys.exit``.

Wheel metadata stays free of URL dependencies either way.
tts._ensure_misaki registers the snapshot under the top-level name only when
no real misaki distribution is installed. Revisit both fixes when upstream
releases.
"""
