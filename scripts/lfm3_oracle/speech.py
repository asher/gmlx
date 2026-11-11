"""Record the LFM3-Omni speech-output reference that
tests/models/test_lfm3_speech_live.py checks gmlx against. Runs in an
environment with torch and transformers, on a folder holding the
LiquidAI/LFM3-Omni-2B checkpoint, its codec and its code. The output
holds arrays only.

    python scripts/lfm3_oracle/speech.py <model_dir> tests/fixtures/lfm3_oracle/speech.npz

It records, in float32 on the CPU:

- the depth decoder's logits for two hidden states and two frames of
  codes, with every codebook masked and with a fixed half of them
  masked;
- the embedding the two frames feed back to the text model;
- the codec decoder's audio for eight frames of codes.

It then records the same from the checkpoint in bfloat16, the dtype it
stores (``*_bf16``). That is the reference's own 16-bit result.

The hidden states and the codes are seeded noise: these are checks of
the functions, not of speech.
"""

from __future__ import annotations

import copy
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

ROWS, FRAMES = 2, 8
HIDDEN_SCALE = 2.0


def main() -> None:
    model_dir, out_path = sys.argv[1], sys.argv[2]
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, dtype=torch.float32).eval()
    config = model.config
    books, eos = config.num_codebooks, config.audio_eos_code

    rng = np.random.default_rng(7)
    hidden = (HIDDEN_SCALE * rng.standard_normal(
        (ROWS, config.text_config.hidden_size))).astype(np.float32)
    codes = rng.integers(0, eos, size=(ROWS, books))
    masked_some = rng.random((ROWS, books)) < 0.5
    codec_codes = rng.integers(0, eos, size=(FRAMES, books))
    out: dict = {
        "hidden": hidden, "codes": codes, "masked_some": masked_some,
        "codec_codes": codec_codes,
    }

    def depth(dtype, suffix, store):
        with torch.no_grad():
            h = torch.from_numpy(hidden).to(dtype)
            c = torch.from_numpy(codes)
            for name, mask in (("all", np.ones_like(masked_some)),
                               ("some", masked_some)):
                logits = model.depth_decoder(h, c, torch.from_numpy(mask))
                out[f"logits_{name}{suffix}"] = (
                    logits.float().numpy().astype(store))
            out[f"feedback{suffix}"] = model.get_audio_out_embeds(
                c).float().numpy().astype(store)

    depth(torch.float32, "", np.float32)
    model.to(torch.bfloat16)
    depth(torch.bfloat16, "_bf16", np.float16)

    codec = processor.audio_tokenizer.eval()
    grid = torch.from_numpy(codec_codes).T[None]
    with torch.no_grad():
        out["audio_bf16"] = codec.decode(
            grid, return_dict=True).audio_values[0].float().numpy()
        exact = copy.deepcopy(codec).float()
        out["audio"] = exact.decode(
            grid, return_dict=True).audio_values[0].float().numpy()
    out["codec_dtype"] = np.asarray(str(next(codec.parameters()).dtype))

    np.savez_compressed(out_path, **out)
    for key, value in out.items():
        print(f"{key}: {value.shape} {value.dtype}")


if __name__ == "__main__":
    main()
