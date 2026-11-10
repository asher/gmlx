"""Record a spoken sentence from the LFM3-Omni reference, for
tests/models/test_lfm3_speech_live.py to replay through gmlx. Runs in an
environment with torch and transformers, on a folder holding the
LiquidAI/LFM3-Omni-2B checkpoint and its code. The output holds arrays
only.

    python scripts/lfm3_oracle/tts.py <model_dir> tests/fixtures/lfm3_oracle/tts.npz

The reference speaks one sentence with the card's speech prompt, in
float32 on the CPU. The frames it sampled are then fed back in one
forward pass, so every hidden state that picked a frame is known. The
file holds:

- ``input_ids``: the prompt, ending in the token that starts the speech;
- ``codes``: the frames, ``[frames, codebooks]``;
- ``top1``: the best code of each codebook in the first pass of each
  frame, ``[frames + 1, codebooks]``. The last row follows the last
  frame, where the reference ended the audio;
- ``logits``: the first-pass logits of the frames in ``logit_frames``;
- ``kl_bf16`` and ``top1_bf16``: how far the checkpoint in bfloat16, the
  dtype it stores, is from the float32 run on the same frames. The
  divergence is KL(float32 || bfloat16) per frame, averaged over the
  codebooks.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

TEXT = "The children built a small boat and sailed it across the pond."
SYSTEM = "Perform TTS in American English using a female voice."
SEED = 7
LOGIT_FRAMES = 8


def main() -> None:
    model_dir, out_path = sys.argv[1], sys.argv[2]
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, trust_remote_code=True, dtype=torch.float32).eval()
    config = model.config
    inputs = processor.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": TEXT}],
        add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt")
    torch.manual_seed(SEED)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=256, generation_mode="audio",
            audio_temperature=0.4, audio_repetition_penalty=1.2)
    count = int(outputs.audio_lengths[0])
    codes = outputs.audio_codes[0, :, :count].T.contiguous()
    prompt = torch.cat((
        inputs["input_ids"],
        torch.tensor([[config.audio_start_token_id]])), dim=1)
    replay = torch.cat((
        prompt, torch.full((1, count), config.audio_out_token_id)), dim=1)
    start = prompt.shape[1] - 1
    masked = torch.ones((count + 1, config.num_codebooks), dtype=torch.bool)
    blank = torch.zeros((count + 1, config.num_codebooks), dtype=torch.long)

    def first_pass(dtype) -> torch.Tensor:
        with torch.no_grad():
            hidden = model.model(
                input_ids=replay, audio_out_codes=codes).last_hidden_state
            hidden = hidden[0, start:start + count + 1].to(dtype)
            return model.depth_decoder(hidden, blank, masked).float()

    exact = first_pass(torch.float32)
    model.to(torch.bfloat16)
    rounded = first_pass(torch.bfloat16)
    log_p = torch.log_softmax(exact.double(), dim=-1)
    log_q = torch.log_softmax(rounded.double(), dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1).mean(-1)
    agree = (exact.argmax(-1) == rounded.argmax(-1)).double().mean(-1)
    frames = np.linspace(0, count, LOGIT_FRAMES).round().astype(np.int64)
    out = {
        "input_ids": prompt[0].numpy().astype(np.int32),
        "codes": codes.numpy().astype(np.int16),
        "top1": exact.argmax(-1).numpy().astype(np.int16),
        "logit_frames": frames,
        "logits": exact[frames].numpy().astype(np.float16),
        "kl_bf16": kl.numpy().astype(np.float32),
        "top1_bf16": agree.numpy().astype(np.float32),
    }
    np.savez_compressed(out_path, **out)
    for key, value in out.items():
        print(f"{key}: {value.shape} {value.dtype}")
    print(f"frames {count}  bf16 KL mean {float(kl.mean()):.5f} "
          f"top-1 agreement {float(agree.mean()):.4f}  "
          f"ends the audio: {bool((exact[-1].argmax(-1) == config.audio_eos_code).any())}")


if __name__ == "__main__":
    main()
