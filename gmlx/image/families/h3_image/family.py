"""MiniMax-H3-Image fingerprint, constants, config synthesis and sampling.

Shapes are GGUF order, input dim first. The file's tensor names are
ComfyUI's names for the curve-form DiT that unsloth/MiniMax-H3-Image-GGUF
ships: MiniMax-H3's DiT without the audio stream, whose video head stacks
one head per step of the distilled schedule (PDD), each 24 x 2 x 2 wide.
Head 0 is a full head and heads 1 to 5 are offsets from it.

The default schedule is the grid itself: six steps on the sigmas of the
release's pdd_config.json, at guidance 1. A step reads the heads of the
grid intervals it spans (``step_heads``).
"""

from __future__ import annotations

import re

import numpy as np

from gmlx.image.fingerprint import HeaderView, ImageModelError, Match, Refusal
from gmlx.image.h3.anchors import AnchorRule
from gmlx.image.h3.encoder import spec
from gmlx.image.h3.layout import Timeline
from gmlx.image.registry import (ConstantsRow, EditParts, Family, SamplingRow,
                                 VariantRow)

TITLE = "MiniMax-H3-Image"

# Width, heads, head dim, feed-forward width, text width, latent channels,
# patch, adaLN table rows and basis, RoPE frequencies per axis.
_D, _H, _HD, _F, _T = 5376, 56, 128, 14336, 5120
_C, _PATCH = 24, (1, 2, 2)
_GRID, _BASIS, _FREQS = 1025, 8, 16
_DEPTH, _REFINER = 50, 2
_INNER = _H * _HD
_PD = _C * _PATCH[0] * _PATCH[1] * _PATCH[2]

_REPO = "MiniMaxAI/MiniMax-H3-Image"
_REVISION = "5f90473023975e6271b0ed1dd84c4e88e4bbc6d3"
# pdd_config.json: the grid the heads were distilled on, one head a step.
PDD_SIGMAS = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75,
              0.5327739119529724, 0.23200887441635132, 0.0)
STEPS = len(PDD_SIGMAS) - 1

_TOP = {
    "video_patch_proj.weight": (_PD, _D),
    "video_patch_proj.bias": (_D,),
    "condition_proj.weight": (_T, _D),
    "condition_proj.bias": (_D,),
    "adaln_t_table": (_BASIS, _GRID),
    "rope.inv_freq": (_FREQS,),
    "token_refiner.final_norm.weight": (_D,),
    "final_layer.norm.weight": (_D,),
    "final_layer.adaln_proj.linear.weight": (_BASIS, 2 * _D),
    "final_layer.adaln_proj.linear.bias": (2 * _D,),
    "final_layer.video_out.weight": (_D, STEPS * _PD),
    "final_layer.video_out.bias": (STEPS * _PD,),
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

# transformer/config.json values no tensor shape determines: the time
# embedder the curve table replaces, the audio stream the release turns
# off, and the norms; then the pipeline's image sizes and the PDD grid.
CONSTANTS = ConstantsRow(
    values={
        "patch_size": list(_PATCH),
        "timestep_input_dim": 256,
        "time_embed_hidden_size": _D,
        "time_embed_dim": 2688,
        "audio_latents_dim": 32,
        "audio_enabled": False,
        "norm_eps": 1e-05,
        "qk_norm_eps": 1e-05,
        "final_norm_eps": 1e-05,
        "vae_scale_factor": 16,
        "size_multiple": 32,
        "pdd_sigmas": list(PDD_SIGMAS),
    },
    repo=_REPO,
    revision=_REVISION,
)

# The packed sequence: text, then each reference image as one latent
# frame at its own time after the text, then the target image as one
# latent frame on H3's clock after them, patchified (1, 2, 2); spatial
# positions span GRID_SCALE over each frame's square-root area. Text and
# target run at the step's time, reference images at least at REF_TIME.
TIMELINE = Timeline((1, 4, 4, 4, 4), 5.0 / 3.0, 32.0, _PATCH[1],
                    ("text", "anchors", "audio", "video"))
REF_TIME = 0.999
# Reference rows mix as video's keyframe rows do (comfy model.py:567-580):
# 0.999 of the clean rows and 0.001 of noise from a generator every
# reference restarts at the request's seed, at a time of at least 0.999.
REF_RULE = AnchorRule(aug=REF_TIME, seed_offset=0)
# The reference images of an edit, by ComfyUI's
# MiniMaxH3ImageConditioning (comfy_extras/nodes_minimax_h3.py): image_1
# is the picture to edit, the VAE's copy of each has its short edge scaled
# into 1024 to 2048 pixels and the vision tower's to 1024, each side on
# the 32 pixel grid, by Lanczos, and the prompt reads each after its
# label at the image processor's bounds in pixels. A request with
# references and no size takes image_1's VAE size. gmlx takes up to 16.
REFERENCES = ConstantsRow(
    values={
        "max_refs": 16,
        "short_edge": 1024,
        "vae_max_short_edge": 2048,
        "grid": 32,
        "label": "Picture {i}: ",
        "pixels": [3136, 12845056],
    },
    repo="github.com/comfyanonymous/ComfyUI",
    revision="2aa4d535a76e4b6acf6a94f263bc482430c396dd",
)
# Modality tags of the adaLN rows, as H3's: image, text. A prompt's vision
# rows take the image tag.
TAGS = {"video": 0, "text": 1}
ATTENTION_CHUNK = 4096

IDENTITY_TENSORS = ("adaln_t_table", "rope.inv_freq",
                    "final_layer.norm.weight",
                    "token_refiner.final_norm.weight")
_PRUNED = VariantRow("pruned", "unsloth/MiniMax-H3-Image-GGUF",
                     "2eda8f761a1625818e2933e7e726e337d197bc27",
                     ("minimax_h3_image_pruned-UD-Q4_K_XL.gguf",))
VARIANTS = {
    "09989a5803d8d2cd0cf236a4c9a73d48ab2cfbac06c72812e3bad034bb6a0c12":
        _PRUNED,
}
# The release's size, 2048x2048 pixels, which the model is tuned for.
SAMPLING = {
    "pruned": SamplingRow(STEPS, 1.0, 1.0, _REPO, _REVISION,
                          sigmas=PDD_SIGMAS[:-1], size=(2048, 2048)),
}


def model_sigma(sigma) -> float:
    """The sigma the release's DiT reads: the timestep sigma x 1000 in
    float32, over 1000."""
    return float(np.float32(sigma) * np.float32(1000)) / 1000.0


def grid_point(sigma: float) -> int:
    """The PDD grid point nearest ``sigma``, the first of a tie."""
    return min(range(len(PDD_SIGMAS)),
               key=lambda i: abs(PDD_SIGMAS[i] - sigma))


def step_heads(sigma, target) -> tuple:
    """The PDD heads a step from ``sigma`` to ``target`` reads, by
    ComfyUI's rule: start is the grid point nearest the sigma the DiT
    reads, at most the last step's, and stop the one nearest the target,
    at least one past start. Returns start, stop and the float32 weight of
    each grid interval from start to stop, its share of the span. Head 0
    is the full head and heads 1 to 5 offsets, so the step reads head 0
    plus each head past it by its weight."""
    start = min(grid_point(model_sigma(sigma)), STEPS - 1)
    stop = max(grid_point(float(np.float32(target))), start + 1)
    dt = [PDD_SIGMAS[k] - PDD_SIGMAS[k + 1] for k in range(start, stop)]
    w = np.asarray(dt, np.float32) / np.float32(sum(dt))
    return start, stop, tuple(float(v) for v in w)


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
    if view.has("audio_patch_proj.weight") or not (
            view.has("video_patch_proj.weight")
            and view.has("condition_proj.weight")):
        return None
    if view.has("time_embedder.proj_in.weight"):
        return Refusal(
            f"a {TITLE} GGUF with the full time embedder: gmlx image runs "
            f"the curve-table form that unsloth/MiniMax-H3-Image-GGUF ships")
    scaled = [n for n in view.tensors if n.endswith(".weight_scale")]
    if scaled:
        return Refusal(
            f"a {TITLE} GGUF with int8 linears and separate weight_scale "
            f"tensors ({len(scaled)}), which gmlx image does not read: use "
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
        return Refusal(f"a {TITLE} relative that gmlx image does not "
                       f"support: {shown}{more}")
    return Match()


def synth_config(view: HeaderView) -> dict:
    """The keys of MiniMaxAI's transformer/config.json, from shapes and
    constants, plus the curve table's size and the PDD heads."""
    c = CONSTANTS.values
    s = view.dims
    patch = c["patch_size"]
    pixels = patch[0] * patch[1] * patch[2]
    head_dim = s("blocks.0.attn.q_norm.weight")[0]
    hidden = s("video_patch_proj.weight")[1]
    latents = s("video_patch_proj.weight")[0] // pixels
    heads, rem = divmod(s("final_layer.video_out.weight")[1], latents * pixels)
    if rem or heads != len(c["pdd_sigmas"]) - 1:
        raise ImageModelError(
            f"{TITLE} runs one head for each of its "
            f"{len(c['pdd_sigmas']) - 1} steps, and this video_out holds "
            f"{s('final_layer.video_out.weight')[1]} rows")
    return {
        "hidden_size": hidden,
        "num_layers": view.count("blocks"),
        "token_refiner_num_layers": view.count("token_refiner.blocks"),
        "num_attention_heads": s("blocks.0.attn.qkv_proj.weight")[1]
        // (3 * head_dim),
        "attention_head_dim": head_dim,
        "ffn_hidden_size": s("blocks.0.mlp.fc1.weight")[1] // 2,
        "latents_dim": latents,
        "audio_latents_dim": c["audio_latents_dim"],
        "patch_size": list(patch),
        "text_dim": s("condition_proj.weight")[0],
        "timestep_input_dim": c["timestep_input_dim"],
        "time_embed_hidden_size": c["time_embed_hidden_size"],
        "time_embed_dim": c["time_embed_dim"],
        "adaln_out_features": s("blocks.0.adaln_proj.linear.weight")[1],
        "final_adaln_out_features":
            s("final_layer.adaln_proj.linear.weight")[1],
        "rope_inv_freq_len": s("rope.inv_freq")[0],
        "norm_eps": c["norm_eps"],
        "qk_norm_eps": c["qk_norm_eps"],
        "final_norm_eps": c["final_norm_eps"],
        "audio_enabled": c["audio_enabled"],
        "adaln_grid": s("adaln_t_table")[1],
        "adaln_basis": s("adaln_t_table")[0],
        "pdd_steps": heads,
        "pdd_sigmas": list(c["pdd_sigmas"]),
    }


def build(config: dict):
    from .model import H3ImageDiT

    return H3ImageDiT(config)


def _edit(name: str):
    def call(*args, **kwargs):
        from . import edit

        return getattr(edit, name)(*args, **kwargs)

    call.__name__ = name
    return call


# H3 video's int8 linears less the VSA gate, which this DiT lacks: the
# four linears of each block, fc2 with one activation scale per 1024 of
# its input columns.
INT8_LINEARS = {"blocks.*.attn.qkv_proj": None,
                "blocks.*.attn.out_proj": None,
                "blocks.*.mlp.fc1": None,
                "blocks.*.mlp.fc2": 1024}


FAMILY = Family(
    name="h3_image",
    title=TITLE,
    order=50,
    arch_strings=frozenset({None}),
    # HunyuanVideo GGUFs carry hyvid too.
    shared_arch_strings=frozenset({"hyvid"}),
    strip_prefixes=("model.diffusion_model.", "diffusion_model."),
    fingerprint=fingerprint,
    synth_config=synth_config,
    constants=CONSTANTS,
    build=build,
    encoder=spec,
    fp32_keep=("adaln_t_table", "adaln_proj.", "rope.inv_freq",
               "video_patch_proj.", "final_layer.video_out."),
    identity_tensors=IDENTITY_TENSORS,
    variants=VARIANTS,
    unknown_variant="pruned",
    sampling=SAMPLING,
    # Every file runs the PDD heads, distilled to guidance 1.
    unguided={"pruned": "", "unknown": ""},
    ref_encode_seed=42,
    # The release's sizes: 655,360 to 20,000,000 pixels at 1:3 to 3:1.
    # 7712 is the longest side on the 32 pixel grid inside all three:
    # 7712x2592 is 19,989,504 pixels at 2.975:1.
    max_pixels=20_000_000,
    max_side=7712,
    min_pixels=655_360,
    aspect=((1, 3), (3, 1)),
    int8_linears=INT8_LINEARS,
    int8_attention=True,
    edit=EditParts(max_refs=REFERENCES.values["max_refs"],
                   prepare_all=_edit("prepare"), size=_edit("default_size"),
                   token_ids=_edit("token_ids"), encode=_edit("encode"),
                   template_id=_edit("template_id"), render=_edit("render")),
)
