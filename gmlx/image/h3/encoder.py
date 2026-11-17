"""MiniMax-H3's text encoder spec, shared by video and image: the
Qwen3-VL-32B-Instruct pair, its tap, its added tokens and its vision
tokens."""

from __future__ import annotations

TEXT_WIDTH = 5120

# The text encoder: the unmodified Qwen3-VL-32B-Instruct, read after
# layer 50 of 64 with no final norm, on the raw prompt. The stock unsloth
# pair is loaded as published; layers 51 to 64 are dropped after the load.
ENCODER_IDENTITY = (
    ("blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight",
     "8531f22e9de8fe531d0e360411bd87c5677b27b07eb0784f4f98189372a07b8b"),
    ("blk.49.ffn_norm.weight",
     "model.layers.49.post_attention_layernorm.weight",
     "af0ad530df7d2b61f7ddc80b2d59781e070f7a40517646ac68e9fb9e51e3882d"),
    ("output_norm.weight", "model.norm.weight",
     "4d58c8cf1e5f169233b59956525509760c7d208646f1343b5b588ef708410492"),
)
ENCODER_IDENTITY_SOURCE = (
    "unsloth/Qwen3-VL-32B-Instruct-GGUF at "
    "b9262a359f54dead8e2609f6146e2fc3398fd0d9, "
    "Qwen3-VL-32B-Instruct-Q4_K_M.gguf with sha256 "
    "92d605566f8661b296251c535ed028ecf81c32e14e06948a3d8bef829e96a804")
MMPROJ_IDENTITY = (
    ("v.blk.0.ln1.weight", "vision_tower.blocks.0.norm1.weight",
     "5a1b2187705294ecd1c3710c914d604c7a0345b7910437cbda9fb0428c136707"),
    ("v.blk.26.ln2.weight", "vision_tower.blocks.26.norm2.weight",
     "1de6af52344a84eb416ba220ce86ec8fda190426463c7b8d02778f5af2f41ca7"),
    ("v.post_ln.weight", "vision_tower.merger.norm.weight",
     "a2a0329d7b6f68319a03157c16148dcdf96963cb5964f2e1634755d2690d17f0"),
)
MMPROJ_IDENTITY_SOURCE = (
    "unsloth/Qwen3-VL-32B-Instruct-GGUF at "
    "b9262a359f54dead8e2609f6146e2fc3398fd0d9, mmproj-BF16.gguf with sha256 "
    "f42400deb87085f1e76159a92aedd276050c665c72423597413d341c36c18c71")
TAP_LAYER = 50

# MiniMax's tokenizer adds these to Qwen's at fixed ids; the prompt guides
# write every dialogue and lyric line with them. An empty prompt is the
# pad token alone, as ComfyUI encodes it.
ADDED_TOKENS = (("<d>", 151669), ("</d>", 151670), ("<|cutoff|>", 151671),
                ("<|lyrics_start|>", 151672), ("<|lyrics_end|>", 151673),
                ("<|caption_start|>", 151674), ("<|caption_end|>", 151675))
EMPTY_IDS = (151643,)
# Qwen3-VL's vision markers and image pad; a vision token covers
# VISION_PX x VISION_PX pixels.
VISION_START, VISION_END, IMAGE_PAD = 151652, 151653, 151655
VISION_PX = 32


def vision_tokens(width: int, height: int) -> int:
    """The vision tokens of a keyframe, one per 32 x 32 pixels."""
    return (width // VISION_PX) * (height // VISION_PX)


def spec():
    from gmlx.image.text_encoder import EncoderSpec, IdentityTensor

    return EncoderSpec(
        name="qwen3-vl-32b",
        arch="qwen3vl",
        model_type="qwen3",
        hidden=TEXT_WIDTH,
        layers=64,
        tap_layers=(TAP_LAYER,),
        template="{}",
        identity=tuple(IdentityTensor(*r) for r in ENCODER_IDENTITY),
        identity_source=ENCODER_IDENTITY_SOURCE,
        activation="float32",
        vision_type="qwen3_vl",
        vision_identity=tuple(IdentityTensor(*r) for r in MMPROJ_IDENTITY),
        vision_identity_source=MMPROJ_IDENTITY_SOURCE,
        drop_untapped=True,
        added_tokens=ADDED_TOKENS,
        empty_ids=EMPTY_IDS,
    )
