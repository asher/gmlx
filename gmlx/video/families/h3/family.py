"""MiniMax-H3 fingerprint, constants and config synthesis.

Shapes are GGUF order, input dim first. The file's tensor names are
ComfyUI's names for the curve-form DiT that unsloth/MiniMax-H3-GGUF ships:
the time embedder is replaced by a (1025, 8) table and each adaLN linear
reads its 8 columns. The file has no metadata keys.

The header cannot tell a fused qkv stored [q; k; v] from one interleaved
per head, so a known file is also named by a hash of rows of q, k and v
head 0 in block 0's qkv (``layouts``); any other file loads with a
warning.
"""

from __future__ import annotations

import re

from gmlx.image.fingerprint import HeaderView, Match, Refusal
from gmlx.image.registry import ConstantsRow
from gmlx.video.anchors import AnchorRule
from gmlx.video.layout import Timeline
from gmlx.video.registry import VideoFamily, VideoVariantRow

TITLE = "MiniMax-H3"

# Width, heads, head dim, feed-forward width, text width, latent and audio
# channels, patch, adaLN table rows and basis, RoPE frequencies per axis.
_D, _H, _HD, _F, _T = 5376, 56, 128, 14336, 5120
_C, _CA, _PATCH = 24, 32, (1, 2, 2)
_GRID, _BASIS, _FREQS = 1025, 8, 16
_DEPTH, _REFINER = 50, 2
_INNER = _H * _HD
_PD = _C * _PATCH[0] * _PATCH[1] * _PATCH[2]

_TOP = {
    "video_patch_proj.weight": (_PD, _D),
    "video_patch_proj.bias": (_D,),
    "audio_patch_proj.weight": (_CA, _D),
    "audio_patch_proj.bias": (_D,),
    "condition_proj.weight": (_T, _D),
    "condition_proj.bias": (_D,),
    "adaln_t_table": (_BASIS, _GRID),
    "rope.inv_freq": (_FREQS,),
    "token_refiner.final_norm.weight": (_D,),
    "final_layer.norm.weight": (_D,),
    "final_layer.adaln_proj.linear.weight": (_BASIS, 2 * _D),
    "final_layer.adaln_proj.linear.bias": (2 * _D,),
    "final_layer.video_out.weight": (_D, _PD),
    "final_layer.video_out.bias": (_PD,),
    "final_layer.audio_out.weight": (_D, _CA),
    "final_layer.audio_out.bias": (_CA,),
}
_ATTN_MLP = {
    "norm1.weight": (_D,),
    "norm2.weight": (_D,),
    "attn.qkv_proj.weight": (_D, 3 * _INNER),
    "attn.q_norm.weight": (_HD,),
    "attn.k_norm.weight": (_HD,),
    "attn.out_proj.weight": (_INNER, _D),
    "mlp.fc1.weight": (_D, 2 * _F),
    "mlp.fc2.weight": (_F, _D),
}
_BLOCK = {
    **_ATTN_MLP,
    "adaln_proj.linear.weight": (_BASIS, 6 * 3 * _D),
    "adaln_proj.linear.bias": (6 * 3 * _D,),
}
BLOCK_GROUPS = (("blocks", _DEPTH, _BLOCK),
                ("token_refiner.blocks", _REFINER, _ATTN_MLP))
_GROUP = re.compile(r"(blocks|token_refiner\.blocks)\.(\d+)\.")

CONSTANTS = ConstantsRow(
    values={
        "patch_size": list(_PATCH),
        # The time embedder the curve table replaces: its sinusoid width,
        # hidden width and output width.
        "freq_dim": 256,
        "time_embed_hidden_dim": _D,
        "time_embed_dim": 2688,
        "rope_theta": 10000.0,
        "norm_eps": 1e-05,
        "qk_norm_eps": 1e-05,
        "final_norm_eps": 1e-05,
    },
    repo="MiniMaxAI/MiniMax-H3",
    revision="42ed227ee7df40d41602854ae760620d6eb651fe",
)

# The packed sequence: text, anchors, target audio, target video. Video
# rows patchify (1, 2, 2) over the latent; a latent frame k spans
# FRAME_PER_TOKEN[k % 5] frames on a clock of FRAME_RESCALE per frame;
# spatial positions span GRID_SCALE over the canvas's square-root area.
# Condition rows run pinned near clean.
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0
GRID_SCALE = 32.0
SEGMENT_ORDER = ("text", "anchors", "audio", "video")
TIMELINE = Timeline(FRAME_PER_TOKEN, FRAME_RESCALE, GRID_SCALE, _PATCH[1],
                    SEGMENT_ORDER)
# Anchor rows: video rows take 0.001 of noise from the request seed and
# run at a time of at least 0.999; audio rows take none and run clean.
ANCHOR_RULES = {"video": AnchorRule(aug=0.999, seed_offset=0),
                "audio": AnchorRule(aug=1.0, seed_offset=1)}
# Clip sizing: 24 frames a second, 17 n + 5 frames to 5 n + 2 latent
# frames, 16 pixels a latent pixel, 40 audio latent frames a second.
FPS = 24
AUDIO_LATENT_FPS = 40
VAE_SCALE = 16
CANVAS_MULTIPLE = 32
VIDEO_CHANNELS, AUDIO_CHANNELS = _C, _CA


def align_frames(n: int) -> int:
    """The frame count of at least ``n`` (and at least 5) on the 17 k + 5
    grid."""
    n = max(5, n)
    return n + (5 - n) % 17


def latent_frames(frames: int) -> int:
    return 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2


def audio_frames(frames: int) -> int:
    """Audio latent frames per stereo channel for a clip of ``frames``."""
    return round(frames / FPS * AUDIO_LATENT_FPS)


def latent_shapes(frames: int, width: int, height: int) -> tuple:
    """The video (1, C, T, H, W) and audio (1, C, 2, T) latent shapes of a
    clip. ``frames`` is on the 17 k + 5 grid and the canvas a multiple of
    32."""
    if frames != align_frames(frames):
        raise ValueError(f"{frames} frames is not on the 17 k + 5 grid")
    if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise ValueError(f"{width}x{height} is not a multiple of "
                         f"{CANVAS_MULTIPLE}")
    return ((1, _C, latent_frames(frames), height // VAE_SCALE,
             width // VAE_SCALE), (1, _CA, 2, audio_frames(frames)))


# Modality tags of the adaLN rows: video, text, audio.
TAGS = {"video": 0, "text": 1, "audio": 2}
SHIFTS = {"video": 12.0, "audio": 3.0}
# lightx2v's 768p Turbo LoRAs sample video at shift 6.
TURBO_768_SHIFT = 6.0
ATTENTION_CHUNK = 4096

IDENTITY_TENSORS = ("adaln_t_table", "rope.inv_freq",
                    "final_layer.norm.weight",
                    "token_refiner.final_norm.weight")
LAYOUT_TENSOR = "blocks.0.attn.qkv_proj.weight"
LAYOUT_ROWS = 4
_REPO = "unsloth/MiniMax-H3-GGUF"
_REVISION = "babc040eff0084465460ead58fdb8ec44296f8dd"
_FL2VA = VideoVariantRow("fl2va", _REPO, _REVISION, (
    "minimax_h3_fl2va_pruned-Q2_K.gguf",
    "minimax_h3_fl2va_pruned-Q3_K.gguf",
    "minimax_h3_fl2va_pruned-Q4_K.gguf",
    "minimax_h3_fl2va_pruned-Q5_0.gguf",
    "minimax_h3_fl2va_pruned-Q6_K.gguf",
    "minimax_h3_fl2va_pruned-Q8_0.gguf",
    "minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf",
    "minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf"))
VARIANTS = {
    "6f575c5d2a38a3f69ec31f42aabc6438008497a2032d720a80666e7e549e0414": _FL2VA,
}


def _rung(*files: str, repo: str = _REPO,
          revision: str = _REVISION) -> VideoVariantRow:
    return VideoVariantRow("fl2va", repo, revision, files)


_MOLBAL = ("molbal/MiniMax-H3-GGUF", "b45874371c61c49bf04096602f5527aac71b360b")
# The layout hash of each rung. Every unsloth file stores qkv as
# [q; k; v]; molbal's Q4_0 and Q8_0 files too, whose q, k and v rows match
# unsloth's Q8_0 rows at the same places.
LAYOUTS = {
    "a4f7c235ae2415d7526937b1b91bb8a2b4ea7cef3a72103bee800b210f10364e":
        _rung("minimax_h3_fl2va_pruned-Q2_K.gguf",
              "minimax_h3_fl2va_pruned-UD-Q2_K_XL.gguf"),
    "0eca77848e4e8c23bca389a40de1a64810c5f7d2560a9b9fb8c589c6ab29bad3":
        _rung("minimax_h3_fl2va_pruned-Q3_K.gguf"),
    "5a8e5ef72a9b1df01835139b24ed7bd402680512cef79292340f75d89fcc338f":
        _rung("minimax_h3_fl2va_pruned-Q4_K.gguf"),
    "9df69106f6570217dba54fc74d8ddee9463d94fe0dd7c6b1effa5d37d6180b9b":
        _rung("minimax_h3_fl2va_pruned-Q5_0.gguf"),
    "e2e104e213f7f8c1b1678a023274bef782643e6e77a07a74602c2dbe1149107a":
        _rung("minimax_h3_fl2va_pruned-Q6_K.gguf"),
    "94b5fa969e849d160f296a0b948a232ea270af02f8d2a6460b3110e9bdf79258":
        _rung("minimax_h3_fl2va_pruned-Q8_0.gguf"),
    "139b825085b348330837ba19d3cf02726c9cd7679c52f8f4975a4a05a672408d":
        _rung("minimax_h3_fl2va_pruned-UD-Q3_K_XL.gguf"),
    "0e0c97063045a8edb507f584bab3b2b6489669f521065719d36bece1f2052133":
        _rung("minimax_h3_fl2va_pruned_fp8_Q4_0.gguf", repo=_MOLBAL[0],
              revision=_MOLBAL[1]),
    "e18dc8780ba2d834d30bd594178bc18d628db107d1513d921b83dae081523901":
        _rung("minimax_h3_fl2va_pruned_fp8_Q8_0.gguf", repo=_MOLBAL[0],
              revision=_MOLBAL[1]),
}
REFUSED = {
    "247d976f868cf75afaa2aacee59150489afbc7006da79dec9adfb15d81b580bb": (
        "this is the %s Ref2VA checkpoint (reference images, video or audio); "
        "gmlx video runs the FL2VA checkpoint, and Ref2VA is a follow-up "
        "release" % TITLE),
}


def expected() -> dict:
    """Every tensor name of the file and its shape."""
    out = dict(_TOP)
    for group, count, block in BLOCK_GROUPS:
        for i in range(count):
            out.update({f"{group}.{i}.{n}": s for n, s in block.items()})
    return out


def _past_count(name: str, view: HeaderView) -> bool:
    """Whether a block tensor sits in a block past the family's count."""
    m = _GROUP.match(name)
    if m is None:
        return False
    want = {g: c for g, c, _ in BLOCK_GROUPS}[m[1]]
    return int(m[2]) >= min(want, view.count(m[1]))


def fingerprint(view: HeaderView) -> Match | Refusal | None:
    if not (view.has("video_patch_proj.weight")
            and view.has("audio_patch_proj.weight")):
        return None
    if view.has("time_embedder.proj_in.weight"):
        return Refusal(
            f"a {TITLE} GGUF with the full time embedder: gmlx video runs "
            f"the curve-table form that unsloth/MiniMax-H3-GGUF ships")
    scaled = [n for n in view.tensors if n.endswith(".weight_scale")]
    if scaled:
        return Refusal(
            f"a {TITLE} GGUF with int8 linears and separate weight_scale "
            f"tensors ({len(scaled)}), which gmlx video does not read: use "
            f"a K-quant or Q8_0 file")
    want = expected()
    diffs = [f"{g} has {view.count(g)} blocks, {TITLE} has {c}"
             for g, c, _ in BLOCK_GROUPS if view.count(g) != c]
    diffs += [f"{n} is {view.shape(n)}, {TITLE} has {s}"
              for n, s in want.items()
              if view.has(n) and view.shape(n) != s]
    diffs += [f"{n} is missing" for n in want
              if not view.has(n) and not _past_count(n, view)]
    diffs += [f"{n} is not a {TITLE} tensor" for n in view.tensors
              if n not in want and not _past_count(n, view)]
    if diffs:
        shown = ", ".join(diffs[:8])
        more = f" (+{len(diffs) - 8} more)" if len(diffs) > 8 else ""
        return Refusal(f"a {TITLE} relative that gmlx video does not "
                       f"support: {shown}{more}")
    return Match(layout="unverified")


def synth_config(view: HeaderView) -> dict:
    """The keys of MiniMaxAI's transformer/config.json, from shapes and
    constants, plus the curve table's size."""
    c = CONSTANTS.values
    s = view.dims
    patch = c["patch_size"]
    head_dim = s("blocks.0.attn.q_norm.weight")[0]
    hidden = s("video_patch_proj.weight")[1]
    return {
        "num_attention_heads": s("blocks.0.attn.qkv_proj.weight")[1]
        // (3 * head_dim),
        "attention_head_dim": head_dim,
        "hidden_size": hidden,
        "num_layers": view.count("blocks"),
        "num_refiner_layers": view.count("token_refiner.blocks"),
        "ffn_dim": s("blocks.0.mlp.fc1.weight")[1] // 2,
        "in_channels": s("final_layer.video_out.weight")[1]
        // (patch[0] * patch[1] * patch[2]),
        "audio_in_channels": s("final_layer.audio_out.weight")[1],
        "patch_size": list(patch),
        "text_dim": s("condition_proj.weight")[0],
        "freq_dim": c["freq_dim"],
        "time_embed_hidden_dim": c["time_embed_hidden_dim"],
        "time_embed_dim": c["time_embed_dim"],
        "rope_freq_dim": s("rope.inv_freq")[0],
        "rope_theta": c["rope_theta"],
        "norm_eps": c["norm_eps"],
        "qk_norm_eps": c["qk_norm_eps"],
        "final_norm_eps": c["final_norm_eps"],
        "adaln_grid": s("adaln_t_table")[1],
        "adaln_basis": s("adaln_t_table")[0],
    }


def _build(config: dict):
    from .model import H3Transformer

    return H3Transformer(config)


FAMILY = VideoFamily(
    name="h3",
    title=TITLE,
    order=10,
    arch_strings=frozenset({None, "minimax_h3"}),
    strip_prefixes=("model.diffusion_model.", "diffusion_model."),
    fingerprint=fingerprint,
    synth_config=synth_config,
    constants=CONSTANTS,
    build=_build,
    # The curve table, its adaLN linears, RoPE and the patch and output
    # projections run in float32, as the reference computes them.
    fp32_keep=("adaln_t_table", "adaln_proj.", "rope.inv_freq",
               "video_patch_proj.", "audio_patch_proj.",
               "final_layer.video_out.", "final_layer.audio_out."),
    identity_tensors=IDENTITY_TENSORS,
    variants=VARIANTS,
    refused=REFUSED,
    layouts=LAYOUTS,
    layout_tensor=LAYOUT_TENSOR,
    layout_rows=LAYOUT_ROWS,
    timeline=TIMELINE,
    shifts=SHIFTS,
    anchor_rules=ANCHOR_RULES,
)
