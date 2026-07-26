"""Vendored third-party packages (see THIRD_PARTY_NOTICES.md).

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
