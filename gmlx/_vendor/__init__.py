"""Vendored third-party packages (see THIRD_PARTY_NOTICES.md).

misaki: Kokoro's G2P, snapshotted at hexgrad/misaki@fba1236 (0.9.4 + the
unreleased Python-3.13 support). PyPI's misaki caps at <3.13 and its [en]
extra drags torch/transformers for a transformer mode Kokoro never enables;
the snapshot removes both problems while keeping wheel metadata free of URL
dependencies. tts._ensure_misaki registers it under the top-level name only
when no real misaki distribution is installed. Drop when upstream releases.
"""
