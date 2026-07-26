# Test assets

- `cats.jpg` — the default image for the VLM tier. Sourced from the
  [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) examples (MIT-licensed). A real
  photo exercises the vision encoder far better than a synthetic shapes image.
- `speech.wav` — the default clip for the audio tier: a short spoken phrase
  ("peanut butter and jelly sandwich"), 16 kHz mono PCM, synthesized with
  macOS `say` (no external source). Real speech exercises the audio encoder
  where a tone sweep would not.

Override with `--image PATH` or `$GMLX_E2E_IMAGE`.
