"""Text encoder tap for image conditioning.

An image family conditions its DiT on the hidden states of a chat model
as the text loader builds it, which for Qwen3 is the copied mlx-lm class.
The states are read after a fixed layer and before the final norm. ``tap``
walks the loaded model's own layer list and stops there. It changes and
subclasses no class of the model.

A family whose reference reads the last hidden state after the final
norm, as MOSS-Image reads Qwen3.5, sets ``final_norm``. Its features come
from the decoder stack's own call on a fresh cache, which builds the
masks of its attention and recurrent layers, and not from ``tap``, whose
block calls pass the one causal mask that only attention layers take.

Each prompt is rendered into the family's fixed template, which the family
owns and never reads from the GGUF chat template, and truncated to the
family's token cap. It is encoded alone at batch 1 with no padding, so the
layers take the causal mask and no attention mask is built. With causal
attention and relative rotary positions, the real tokens of a padded and
masked reference batch take the same values up to rounding. A family
whose template opens with a fixed turn drops the rows of that turn from
the conditioning, after they ran: the rows it keeps attended to them.

A family whose reference keeps the rows of right padding in the
conditioning names its pad token, the multiple the rows are padded to
and the rotary position of the pad rows: one position for all of them,
or for each its own index after the real tokens. In the reference each
pad row attends to every real token and to no pad row, itself included.
Pad rows at one position then hold the same values in every layer, so
the tap runs one pad row after the real tokens, with that mask and that
position, and repeats it. Pad rows at their own indices differ, and the
tap runs each of them with that mask.

A family that conditions on reference images loads the encoder with its
vision tower, from the encoder GGUF and its mmproj, as the multimodal
class of the model. A text prompt taps that class by the same walk, and
the family's own code runs a prompt with images. The text class of such
an encoder rotates its queries and keys with the multimodal class's rope
for the tap, so a text prompt gives the same conditioning with and
without the tower loaded.

An ``EncoderHandle`` owns the encoder it loaded and releases it on request,
so the DiT and the VAE can use its memory. An owned encoder holds no
output matrix, which a tap never reads. A borrowed resident chat model is
never released and keeps every weight.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import importlib
import json
import os
import threading
import unicodedata
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .fingerprint import ImageModelError, canonical_bf16


@dataclass(frozen=True)
class IdentityTensor:
    """A small norm tensor that identifies the encoder weights, by its GGUF
    name, its path in the loaded model and the sha256 of its bf16 bits."""
    gguf_name: str
    model_path: str
    sha256: str


# GGUF norm roles of a block and their modules under it, by encoder arch.
# output_norm is the stack's final norm.
_NORMS = {
    "qwen35": (("attn_norm", "input_layernorm"),
               ("post_attention_norm", "post_attention_layernorm"),
               ("attn_q_norm", "self_attn.q_norm"),
               ("attn_k_norm", "self_attn.k_norm"),
               ("ssm_norm", "linear_attn.norm")),
}


@dataclass(frozen=True)
class EncoderSpec:
    name: str                        # "qwen3-4b"
    arch: str                        # GGUF general.architecture
    model_type: str                  # model_type of the loaded model
    hidden: int
    layers: int
    tap_layers: tuple                # layers run before each tap
    template: str                    # "{}" marks the prompt
    max_tokens: int | None = None
    identity: tuple = ()             # IdentityTensor rows
    identity_source: str = ""        # repository and revision of the hashes
    drop: int = 0                    # leading rows left out of the features
    empty: str = ""                  # what an empty prompt is rendered as
    activation: str | None = None    # dtype the tap's layers run in, the
                                     # model's own when None
    vision_type: str | None = None   # model_type of the encoder loaded with
                                     # its vision tower, None without one
    vision_identity: tuple = ()      # IdentityTensor rows of its mmproj
    vision_identity_source: str = ""
    keep_tail: int = 0               # last ids the token cap never cuts
    stack: bool = False              # features hold every tap side by side
    pad_multiple: int = 0            # rows padded to a multiple of this,
                                     # at most max_tokens; 0 for no pads
    pad_id: int | None = None        # token id of the pad rows
    pad_position: int | None = 1     # rotary position of every pad row,
                                     # None for each at its own index
    fp32_qkv: bool = False           # q, k and v cast to float32 after
                                     # their products, for the qk norm,
                                     # the rotary and the attention
    final_norm: bool = False         # the features are the text model's
                                     # output after its final norm, from
                                     # its own call; tap_layers is (layers,)
    trim: bool = False               # the prompt loses its leading and
                                     # trailing whitespace, as a chat
                                     # template's trim filter takes it
    fp32_norms: bool = False         # an encode runs the RMSNorms of the
                                     # decoder stack on their F32 GGUF
                                     # weights, each norm's output in its
                                     # input dtype; the loaded model keeps
                                     # its own weights

    def __post_init__(self):
        if self.fp32_norms and self.arch not in _NORMS:
            raise ValueError(f"{self.name}: no F32 norm map for arch "
                             f"{self.arch!r}")
        if not self.final_norm:
            return
        if tuple(self.tap_layers) != (self.layers,):
            raise ValueError(f"{self.name} reads after the final norm, so "
                             f"its tap is after layer {self.layers} alone")
        if (self.drop or self.stack or self.pad_multiple or self.fp32_qkv
                or self.activation is not None):
            raise ValueError(f"{self.name} reads after the final norm, "
                             "which takes no dropped rows, stacked taps, "
                             "pads, float32 qkv or activation dtype")

    @property
    def template_id(self) -> str:
        """sha256 of what turns a prompt into the rows of a conditioning
        beside the token ids: the template, and the count of dropped rows
        when the spec drops any, that the features stack the taps when
        they do, and the pad rule when the rows are padded."""
        text = self.template
        if self.drop:
            text += f"\x00drop={self.drop}"
        if self.stack:
            text += "\x00stack"
        if self.pad_multiple:
            where = ("each" if self.pad_position is None
                     else self.pad_position)
            text += f"\x00pad={self.pad_multiple},{self.pad_id},{where}"
        if self.final_norm:
            text += "\x00final_norm"
        if self.trim:
            text += "\x00trim"
        if self.fp32_norms:
            text += "\x00fp32_norms"
        return hashlib.sha256(text.encode()).hexdigest()


# Z-Image conditions on Qwen3-4B after layer 35 of 36, which is the
# reference's hidden_states[-2]. The template is what the Qwen3 chat
# template renders for one user turn with the generation prompt and
# thinking on.
Z_IMAGE = EncoderSpec(
    name="qwen3-4b",
    arch="qwen3",
    model_type="qwen3",
    hidden=2560,
    layers=36,
    tap_layers=(35,),
    template="<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    max_tokens=512,
    identity=(
        IdentityTensor(
            "blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight",
            "5812f00704f74996f9460b6f78a3fdaafb759d8c1b9d406f3b3515e6376a33ac"),
        IdentityTensor(
            "blk.17.ffn_norm.weight",
            "model.layers.17.post_attention_layernorm.weight",
            "af391acf15715034200066055fee46b986e8b4fc3ea8fca754e2e8e48f61f8b4"),
        IdentityTensor(
            "output_norm.weight", "model.norm.weight",
            "2e185da56e34e5461e40b8cad00745efc88e6edb956d3879adcba96caaabbbef"),
    ),
    identity_source=("unsloth/Qwen3-4B-GGUF at "
                     "22c9fc8a8c7700b76a1789366280a6a5a1ad1120"),
)

# The encoder each family conditions on, by family name.
ENCODERS = {"z_image": Z_IMAGE}


class EncoderMismatchError(ImageModelError):
    """An encoder that is not the one the family conditions on."""


def encoder_spec(family) -> EncoderSpec:
    """The encoder spec of a family, given as a name or a ``Family``: the
    one its registry row gives, else the one this module holds for it."""
    if isinstance(family, str):
        from .registry import get_family

        try:
            family = get_family(family)
        except ValueError:
            raise ImageModelError(
                f"image family {family!r} has no text encoder in "
                "gmlx") from None
    if family.encoder is not None:
        return family.encoder()
    try:
        return ENCODERS[family.name]
    except KeyError:
        raise ImageModelError(
            f"image family {family.name!r} has no text encoder in "
            "gmlx") from None


def render(spec: EncoderSpec, prompt: str) -> str:
    """The prompt in the family template, without its leading and
    trailing whitespace for a spec that trims. An empty prompt is rendered
    as the spec's ``empty`` text."""
    head, sep, tail = spec.template.partition("{}")
    if not sep:
        raise ValueError(f"template of {spec.name} has no prompt slot")
    if spec.trim:
        prompt = prompt.strip()
    return head + (prompt or spec.empty) + tail


def token_ids(spec: EncoderSpec, tokenizer, prompt: str) -> list[int]:
    """Token ids of the rendered prompt, cut at the family's token cap. The
    cap applies to the whole rendered text, as in the reference, and a
    spec that keeps a tail cuts the ids before its last ``keep_tail``."""
    ids = list(tokenizer.encode(render(spec, prompt), add_special_tokens=False))
    if spec.max_tokens is not None and len(ids) > spec.max_tokens:
        n = spec.keep_tail
        ids = (ids[: spec.max_tokens - n] + ids[len(ids) - n:] if n
               else ids[: spec.max_tokens])
    if len(ids) <= spec.drop:
        raise ValueError("the rendered prompt has no tokens" if not spec.drop
                         else f"the rendered prompt has {len(ids)} tokens, "
                         f"and the first {spec.drop} are dropped")
    return ids


def pad_rows(spec: EncoderSpec, n: int) -> int:
    """Pad rows after ``n`` token ids: up to the next multiple of the
    spec's pad multiple, at most its token cap. 0 for a spec with no
    pads."""
    if not spec.pad_multiple:
        return 0
    if spec.drop or spec.pad_id is None:
        raise ValueError(f"the pad rule of {spec.name} needs a pad id and "
                         "no dropped rows")
    rows = -(-n // spec.pad_multiple) * spec.pad_multiple
    if spec.max_tokens is not None:
        rows = min(rows, spec.max_tokens)
    return max(0, rows - n)


def rows_at_most(spec: EncoderSpec, prompt: str) -> int:
    """The most rows the features of ``prompt`` hold, from its text alone,
    before a tokenizer loads: the encoders here tokenize the NFC form of
    the rendered text by byte-level BPE, so each token id holds at least
    one byte of it. The NFC form of some characters holds more bytes."""
    text = render(spec, prompt)
    n = max(len(t.encode("utf-8"))
            for t in (text, unicodedata.normalize("NFC", text)))
    if spec.max_tokens is not None:
        n = min(n, spec.max_tokens)
    return max(0, n - spec.drop) + pad_rows(spec, n)


def _inner(model):
    """The decoder stack of a loaded text model, or of the language model
    of a multimodal one."""
    inner = getattr(model, "model", None)
    if inner is None:
        inner = getattr(getattr(model, "language_model", None), "model", None)
    return inner


def _text_stack(model):
    """The embedding and the decoder layer list of a loaded text model,
    or of the language model of a multimodal one."""
    inner = _inner(model)
    embed = getattr(inner, "embed_tokens", None)
    layers = getattr(inner, "layers", None)
    if embed is None or layers is None:
        raise TypeError(
            f"{type(model).__name__} has no model.embed_tokens and "
            "model.layers to tap")
    return embed, layers


def tap(model, ids, *, layers: tuple, activation: str | None = None,
        pad: int = 0, pad_id: int | None = None,
        pad_position: int | None = 1,
        fp32_qkv: bool = False) -> list[mx.array]:
    """Hidden states after running the first ``n`` decoder layers, for each
    ``n`` in ``layers``, with no final norm. ``ids`` is one prompt's token
    ids. Each result is (1, L, hidden) in the model's activation dtype, and
    layers past the last tap never run. With ``activation`` the embeddings
    are cast to that dtype, and the layers run in it by promotion over the
    weights as loaded.

    With ``pad`` the result has ``pad`` more rows of token ``pad_id``, each
    attending to every real token and to no pad row. At a ``pad_position``
    they are one row at that rotary position, repeated. With
    ``pad_position`` None each runs at its own index after the real
    tokens.

    With ``fp32_qkv`` every attention casts its q, k and v products to
    float32, so its qk norm, rotary embedding and attention run in
    float32 whatever dtype a quantized product returns."""
    embed, blocks = _text_stack(model)
    stops = sorted(set(layers))
    if not stops or stops[0] < 1 or stops[-1] > len(blocks):
        raise ValueError(f"tap layers {layers} outside 1..{len(blocks)}")
    if pad and pad_id is None:
        raise ValueError("pad rows need a pad id")
    n = len(ids)
    each = bool(pad) and pad_position is None
    rows = pad if each else min(pad, 1)
    seq = list(ids) + [pad_id] * rows
    x = mx.array(np.asarray(seq, np.int32).reshape(1, -1))
    h = embed(x)
    if activation is not None:
        h = h.astype(getattr(mx, activation))
    if pad:
        allowed = np.tril(np.ones((n + rows, n + rows), bool))
        allowed[:, n:] = False
        mask = mx.array(allowed)
        place = (contextlib.nullcontext() if each
                 else pad_rope(model, n, pad_position))
    else:
        mask = "causal" if n > 1 else None
        place = contextlib.nullcontext()
    out: dict[int, mx.array] = {}
    cast = (float32_qkv(model) if fp32_qkv else contextlib.nullcontext())
    with place, cast:
        for i in range(stops[-1]):
            h = blocks[i](h, mask, None)
            if i + 1 in stops:
                out[i + 1] = h
    if pad and not each:
        for k, v in out.items():
            row = mx.broadcast_to(v[:, n:], (1, pad, v.shape[-1]))
            out[k] = mx.concatenate([v[:, :n], row], axis=1)
    return [out[k] for k in layers]


def final_hidden(model, ids) -> mx.array:
    """The output (1, L, hidden) of the decoder stack after its final
    norm, for one prompt's token ids, from the stack's own call at batch 1
    on a fresh cache, as a chat prefill starts. The call builds its
    attention and recurrent masks, and the positions of a text prompt
    start at 0. The call of the wrapping language model is not used: it
    applies the output head, and a multimodal one keeps the positions of
    its last call, which a resident chat model needs for its own next
    one."""
    inner = _inner(model)
    if inner is None or not callable(inner):
        raise TypeError(f"{type(model).__name__} has no decoder stack to "
                        "call")
    owner = model if hasattr(model, "make_cache") else getattr(
        model, "language_model", None)
    make_cache = getattr(owner, "make_cache", None)
    if make_cache is None:
        raise TypeError(f"{type(model).__name__} makes no cache")
    x = mx.array(np.asarray(ids, np.int32).reshape(1, -1))
    return inner(x, cache=make_cache())


class _Float32Out:
    """A product whose output is cast to float32."""

    def __init__(self, product):
        self.product = product

    def __call__(self, x: mx.array) -> mx.array:
        return self.product(x).astype(mx.float32)


_QKV = ("q_proj", "k_proj", "v_proj")


@contextlib.contextmanager
def float32_qkv(model):
    """Run a tap with each attention's q, k and v products cast to
    float32, and restore the products after."""
    _, blocks = _text_stack(model)
    found = []
    for b in blocks:
        attn = getattr(b, "self_attn", None)
        products = [getattr(attn, name, None) for name in _QKV]
        if any(p is None for p in products):
            raise TypeError(f"{type(b).__name__} has no self_attn q_proj, "
                            "k_proj and v_proj to cast")
        found.append((attn, products))
    try:
        for attn, products in found:
            for name, p in zip(_QKV, products):
                setattr(attn, name, _Float32Out(p))
        yield
    finally:
        for attn, products in found:
            for name, p in zip(_QKV, products):
                setattr(attn, name, p)


@functools.lru_cache(maxsize=4)
def _gguf_norms(path: str, arch: str, stamp: tuple) -> tuple:
    """``(module path, F32 weight)`` of each norm _NORMS maps for ``arch``
    in the GGUF at ``path``, the path under the decoder stack. ``stamp``
    is the file's size and mtime, so a changed file reads again."""
    roles = dict(_NORMS[arch])
    out = []
    for name, t in _header_view(path).tensors.items():
        parts = name.split(".")
        if name == "output_norm.weight":
            where = "norm"
        elif (len(parts) == 4 and parts[0] == "blk" and parts[3] == "weight"
              and parts[2] in roles):
            where = f"layers.{parts[1]}.{roles[parts[2]]}"
        else:
            continue
        if t.codec != "F32" or t.path is None:
            raise EncoderMismatchError(
                f"{path}: {name} is {t.codec}, not the F32 norm weight the "
                "encoder reads")
        fd = os.open(t.path, os.O_RDONLY)
        try:
            raw = os.pread(fd, t.nbytes, t.offset)
        finally:
            os.close(fd)
        out.append((where, np.frombuffer(raw, np.float32).copy()))
    return tuple(out)


def _module(inner, where: str):
    obj = inner
    for part in where.split("."):
        if obj is None:
            break
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part, None)
    return obj


def read_norms(path: str, model, spec: EncoderSpec) -> dict:
    """The F32 norm weights of the encoder GGUF at ``path`` by module path
    under the decoder stack of ``model``. Each must round to the weight
    the model holds, so they are the model's own norms."""
    st = os.stat(path)
    inner = _inner(model)
    out = {}
    for where, a in _gguf_norms(os.path.abspath(path), spec.arch,
                                (st.st_size, st.st_mtime_ns)):
        mod = _module(inner, where)
        held = getattr(mod, "weight", None)
        w = mx.array(a)
        if (not isinstance(held, mx.array) or held.shape != w.shape
                or not mx.array_equal(w.astype(held.dtype), held).item()):
            raise EncoderMismatchError(
                f"{path}: the norm at {where} is not the loaded encoder's")
        out[where] = w
    if not out:
        raise EncoderMismatchError(f"{path}: no norm weights to read")
    return out


class _Float32RMSNorm(nn.RMSNorm):
    """An RMSNorm on an F32 weight whose output keeps its input dtype."""

    def __init__(self, weight: mx.array, eps: float):
        super().__init__(weight.shape[0], eps=eps)
        self.weight = weight

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps).astype(x.dtype)


# Models whose norms float32_norms has swapped: id of the decoder stack
# to [entries, norms, swaps]. The first entry swaps, the last exit
# restores, so nested or interleaved entries leave the model's own.
_NORM_SWAPS: dict = {}
_NORM_LOCK = threading.Lock()


def _swap_norms(inner, norms: dict) -> list:
    swaps = []
    for where, w in norms.items():
        head, _, attr = where.rpartition(".")
        parent = _module(inner, head) if head else inner
        mod = getattr(parent, attr)
        if isinstance(mod, nn.RMSNorm):
            swaps.append((parent, attr, mod, _Float32RMSNorm(w, mod.eps)))
        else:
            swaps.append((mod, "weight", mod.weight, w))
    try:
        for obj, attr, _, new in swaps:
            setattr(obj, attr, new)
    except BaseException:
        _restore_norms(swaps)
        raise
    return swaps


def _restore_norms(swaps: list) -> None:
    for obj, attr, old, _ in reversed(swaps):
        setattr(obj, attr, old)


@contextlib.contextmanager
def float32_norms(model, norms: dict):
    """Run an encode with each norm of ``norms`` on its F32 weight, and
    restore the model's own norms after. An RMSNorm is swapped for one
    whose output keeps its input dtype; another norm, such as a gated
    one whose output takes the dtype of its input, keeps its module and
    gets the weight. An entry while the model is swapped shares the swap,
    which the last exit restores; one with other norms is refused."""
    inner = _inner(model)
    key = id(inner)
    with _NORM_LOCK:
        held = _NORM_SWAPS.get(key)
        if held is None:
            _NORM_SWAPS[key] = [1, norms, _swap_norms(inner, norms)]
        elif held[1] is not norms and (
                held[1].keys() != norms.keys()
                or any(held[1][k] is not v for k, v in norms.items())):
            raise RuntimeError("the encoder model already runs on other F32 "
                               "norms")
        else:
            held[0] += 1
    try:
        yield
    finally:
        with _NORM_LOCK:
            held = _NORM_SWAPS[key]
            held[0] -= 1
            if not held[0]:
                del _NORM_SWAPS[key]
                _restore_norms(held[2])


class _PadRope:
    """A rotary embedding that rotates the last of ``n + 1`` rows at the
    pad position and the others as the wrapped one does."""

    def __init__(self, rope, n: int, position: int):
        self.rope, self.n, self.position = rope, n, position

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        if x.shape[-2] != self.n + 1:
            return self.rope(x, offset=offset)
        head = self.rope(x[..., :self.n, :], offset=offset)
        tail = self.rope(x[..., self.n:, :], offset=self.position)
        return mx.concatenate([head, tail], axis=-2)


@contextlib.contextmanager
def pad_rope(model, n: int, position: int):
    """Run a tap of ``n`` tokens and one pad row with each attention's
    rope placing the pad row at ``position``, and restore the ropes
    after."""
    _, blocks = _text_stack(model)
    found = []
    for b in blocks:
        attn = getattr(b, "self_attn", None)
        rope = getattr(attn, "rope", None)
        if rope is None:
            raise TypeError(f"{type(b).__name__} has no self_attn.rope to "
                            "place a pad row with")
        found.append((attn, rope))
    try:
        for attn, rope in found:
            attn.rope = _PadRope(rope, n, position)
        yield
    finally:
        for attn, rope in found:
            attn.rope = rope


class _VisionRope:
    """The multimodal class's rotary embedding in the text class's call:
    one tensor and an offset. A text prompt puts one position on all
    three axes."""

    def __init__(self, dims: int, base: float):
        lang = importlib.import_module(
            "gmlx._vendor.mlx_vlm.models.qwen3_vl.language")
        self.emb = lang.Qwen3VLRotaryEmbedding(dims, base=base)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        n = x.shape[-2]
        pos = mx.tile(mx.arange(offset, offset + n)[None, None], (3, 1, 1))
        return self.emb.apply_rotary(x, x, pos, unsqueeze_dim=1)[0]


def _plain_ropes(model, spec: EncoderSpec) -> list:
    """The attention modules of a text class that the multimodal class of
    ``spec`` would rotate with its own rope, with the plain rope each has.
    Empty for a model with its tower, a spec without one, or a rope with
    scaling."""
    if spec.vision_type is None or getattr(model, "vision_tower", None) \
            is not None:
        return []
    import mlx.nn as nn

    _, blocks = _text_stack(model)
    out = []
    for b in blocks:
        attn = getattr(b, "self_attn", None)
        rope = getattr(attn, "rope", None)
        if type(rope) is not nn.RoPE or rope.traditional or rope.scale != 1.0:
            return []
        out.append((attn, rope))
    return out


@contextlib.contextmanager
def vision_rope(model, spec: EncoderSpec):
    """Run a tap of a text class with the rope of its multimodal class,
    and restore the model's own rope after."""
    found = _plain_ropes(model, spec)
    if not found:
        yield
        return
    swap = _VisionRope(found[0][1].dims, found[0][1].base)
    try:
        for attn, _ in found:
            attn.rope = swap
        yield
    finally:
        for attn, rope in found:
            attn.rope = rope


def _model_param(model, path: str):
    """The tensor at ``path`` in a loaded model. A path under ``model.``
    starts at the decoder stack, which a multimodal or wrapped text model
    holds under its language model."""
    head, _, rest = path.partition(".")
    obj = _inner(model) if head == "model" else getattr(model, head, None)
    for part in rest.split(".") if rest else ():
        if obj is None:
            break
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part, None)
    return obj


def _bf16_digest(arr: mx.array) -> str:
    bits = np.array(arr.astype(mx.bfloat16).view(mx.uint16))
    return hashlib.sha256(bits.tobytes()).hexdigest()


def model_identity(model, spec: EncoderSpec) -> dict[str, str | None]:
    """sha256 of each identity tensor's bf16 bits in a loaded model, by GGUF
    name. A tensor the model lacks maps to None."""
    got: dict[str, str | None] = {}
    for row in spec.identity:
        w = _model_param(model, row.model_path)
        got[row.gguf_name] = (_bf16_digest(w) if isinstance(w, mx.array)
                              else None)
    return got


def _header_view(path: str):
    from gmlx.load.headerscan import scan_gguf
    from gmlx.load.preflight import find_split_shards

    from .fingerprint import view_from_scans

    return view_from_scans(scan_gguf(s) for s in find_split_shards(path))


def file_identity(view, spec: EncoderSpec) -> dict[str, str | None]:
    """The identity digests read from an encoder GGUF, given its header
    view. A tensor the file lacks or stores quantized maps to None."""
    got: dict[str, str | None] = {}
    for row in spec.identity:
        t = view.tensors.get(row.gguf_name)
        if t is None or t.path is None or t.codec not in ("F32", "F16",
                                                          "BF16"):
            got[row.gguf_name] = None
            continue
        fd = os.open(t.path, os.O_RDONLY)
        try:
            raw = os.pread(fd, t.nbytes, t.offset)
        finally:
            os.close(fd)
        got[row.gguf_name] = hashlib.sha256(
            canonical_bf16(raw, t.codec)).hexdigest()
    return got


def _check(spec: EncoderSpec, what: str, model_type, hidden, n_layers,
           identity: dict | None) -> None:
    diffs = []
    if model_type != spec.model_type and model_type != spec.arch:
        diffs.append(f"{what} {model_type!r}, expected {spec.model_type!r}")
    if hidden != spec.hidden:
        diffs.append(f"hidden size {hidden}, expected {spec.hidden}")
    if n_layers != spec.layers:
        diffs.append(f"{n_layers} layers, expected {spec.layers}")
    if diffs:
        raise EncoderMismatchError(
            f"the text encoder is not {spec.name}: " + ", ".join(diffs))
    if identity is None or not spec.identity:
        return
    bad = [r.gguf_name for r in spec.identity
           if identity.get(r.gguf_name) != r.sha256]
    if bad:
        raise EncoderMismatchError(
            f"the text encoder weights are not {spec.name}: "
            f"{', '.join(bad)} differ from {spec.identity_source}. A "
            "fine-tuned encoder needs the explicit encoder override")


def check_encoder_file(view, spec: EncoderSpec, *,
                       override: bool = False) -> None:
    """Refuse an encoder GGUF from its header, before any weights load:
    arch, hidden size and layer count, then the identity digests unless
    ``override`` accepts other weights of the same shape."""
    arch = view.arch
    kv = view.kv
    _check(spec, "arch", arch, kv.get(f"{arch}.embedding_length"),
           kv.get(f"{arch}.block_count"),
           None if override else file_identity(view, spec))


def check_mmproj_file(view, spec: EncoderSpec, *,
                      override: bool = False) -> None:
    """Refuse an mmproj GGUF that is not the vision tower of the spec's
    encoder, from its header before any weights load: a CLIP projector
    to the encoder's hidden size, then the identity digests of its norm
    tensors unless ``override`` accepts other weights of the same
    shape."""
    kv = view.kv
    what = f"the mmproj is not the vision tower of {spec.name}"
    if view.arch != "clip":
        raise EncoderMismatchError(
            f"{what}: its architecture is {view.arch!r}, expected a clip "
            "vision encoder")
    width = kv.get("clip.vision.projection_dim")
    if width != spec.hidden:
        raise EncoderMismatchError(
            f"{what}: it projects to {width}, and the encoder's hidden size "
            f"is {spec.hidden}")
    if override or not spec.vision_identity:
        return
    got = file_identity(view, dataclasses.replace(
        spec, identity=spec.vision_identity))
    bad = [r.gguf_name for r in spec.vision_identity
           if got.get(r.gguf_name) != r.sha256]
    if bad:
        raise EncoderMismatchError(
            f"{what}: {', '.join(bad)} differ from "
            f"{spec.vision_identity_source}. A fine-tuned encoder needs the "
            "explicit encoder override")


def check_encoder(model, config: dict, spec: EncoderSpec, *,
                  override: bool = False) -> None:
    """Refuse a loaded model that is not the family's encoder: its model
    type, hidden size and layer count, then the identity digests of its
    weights unless ``override`` accepts other weights of the same shape,
    such as a fine-tune."""
    _check(spec, "model type", config.get("model_type"),
           config.get("hidden_size"), config.get("num_hidden_layers"),
           None if override else model_identity(model, spec))


@dataclass(frozen=True)
class TextTap:
    """One prompt's encoding: its token ids and one hidden state per tap
    layer, each (1, L, hidden). ``features`` are the rows of the first tap
    that condition the DiT, (L - drop, hidden), or with ``stack`` the rows
    of every tap side by side in tap order, (L - drop, taps * hidden).
    ``pads`` rows of padding follow the rows of the ids."""
    ids: tuple
    hidden: tuple
    template_id: str
    tap_layers: tuple
    drop: int = 0
    stack: bool = False
    pads: int = 0

    @property
    def features(self) -> mx.array:
        if self.stack:
            return mx.concatenate([h[0, self.drop:] for h in self.hidden],
                                  axis=-1)
        return self.hidden[0][0, self.drop:]


def _float_dtype(model) -> str:
    """The dtype name of the model's first floating parameter."""
    from mlx.utils import tree_flatten

    for _, v in tree_flatten(model.parameters()):
        if mx.issubdtype(v.dtype, mx.floating):
            return str(v.dtype).rsplit(".", 1)[-1]
    return ""


class EncoderHandle:
    """A text encoder with its tokenizer and spec. An owned handle can be
    released, a borrowed one cannot."""

    def __init__(self, model, tokenizer, spec: EncoderSpec, *, owned: bool,
                 source_key=None, weights_owner=None, processor=None):
        self.model = model
        self.tokenizer = tokenizer
        # The processor of an encoder loaded with its vision tower.
        self.processor = processor
        self.spec = spec
        self.owned = owned
        self.source_key = source_key
        # What the free-memory reading holds the mapped weights under,
        # when this handle's load registered them. See ``load_encoder``.
        self.weights_owner = weights_owner
        self.dtype_name = spec.activation or _float_dtype(model)
        self.adapters = ""          # text adapter state, in the cache key
        # F32 norm weights an encode runs on, by module path; see
        # EncoderSpec.fp32_norms.
        self.norms: dict = {}

    @property
    def released(self) -> bool:
        return self.model is None

    @property
    def vision(self) -> bool:
        """Whether the encoder holds its vision tower."""
        return getattr(self.model, "vision_tower", None) is not None

    def vision_bytes(self) -> int:
        """Bytes the vision tower holds as loaded, 0 without one."""
        from mlx.utils import tree_flatten

        tower = getattr(self.model, "vision_tower", None)
        if tower is None:
            return 0
        return sum(int(v.nbytes) for _, v in tree_flatten(tower.parameters()))

    def encode(self, prompt: str) -> TextTap:
        if self.model is None:
            raise RuntimeError(
                f"the {self.spec.name} text encoder was released")
        spec = self.spec
        ids = token_ids(spec, self.tokenizer, prompt)
        pads = pad_rows(spec, len(ids))
        norms = (float32_norms(self.model, self.norms) if self.norms
                 else contextlib.nullcontext())
        with vision_rope(self.model, spec), norms:
            if spec.final_norm:
                hidden = [final_hidden(self.model, ids)]
            else:
                hidden = tap(self.model, ids, layers=spec.tap_layers,
                             activation=spec.activation, pad=pads,
                             pad_id=spec.pad_id,
                             pad_position=spec.pad_position,
                             fp32_qkv=spec.fp32_qkv)
            mx.eval(hidden)
        return TextTap(ids=tuple(ids), hidden=tuple(hidden),
                       template_id=spec.template_id,
                       tap_layers=tuple(spec.tap_layers),
                       drop=spec.drop, stack=spec.stack, pads=pads)

    def token_ids(self, prompt: str) -> list[int]:
        return token_ids(self.spec, self.tokenizer, prompt)

    def release(self) -> bool:
        """Drop an owned encoder and return its memory. The tokenizer stays,
        so prompts still map to the keys of cached conditionings. The
        mapped weights leave the free-memory reading with the model, so a
        VAE pass after the release sees the room. Returns False for a
        borrowed or already released encoder, which stays as it is."""
        if not self.owned or self.model is None:
            return False
        self.model = None
        from gmlx.gen.prefill_decay import forget_untracked_weights
        from gmlx.load.loader import collect_after_load

        collect_after_load()
        mx.clear_cache()
        if self.weights_owner is not None:
            forget_untracked_weights(self.weights_owner)
            self.weights_owner = None
        return True


def drop_output_head(model) -> int:
    """Take the output matrix off a text model that has its own, and
    return the bytes it held, 0 for a model without one. A tap stops
    before the final norm, so an owned encoder never reads it, and the
    memory plan counts an encoder without it. A model with tied embeddings
    has none to drop."""
    from mlx.utils import tree_flatten

    if "lm_head" not in model:
        model = getattr(model, "language_model", None)
        if model is None or "lm_head" not in model:
            return 0
    held = sum(int(v.nbytes)
               for _, v in tree_flatten(model["lm_head"].parameters()))
    del model["lm_head"]
    return held


def _load_owned(path: str, mmproj: str | None = None):
    """The text loader's model, config and tokenizer for ``path``, the
    processor, and the owner its mapped weights are registered under for
    the free-memory reading. With ``mmproj`` the model is the multimodal
    class with its vision tower, and the processor is its own; else the
    processor is None. The owner is None when a window of another owner
    is open, which then forgets the weights when it ends."""
    from gmlx.gen import prefill_decay as pd

    owner = None
    if pd.untracked_weights_owner() is None:
        owner = object()
        pd.set_untracked_weights_owner(owner)
    try:
        if mmproj is None:
            from gmlx.load.loader import load_model

            return (*load_model(path), None, owner)
        from gmlx.load.vlm import load_vlm_model

        model, config, processor, tokenizer = load_vlm_model(
            path, mmproj, return_tokenizer=True)
        return model, config, tokenizer, processor, owner
    finally:
        if owner is not None:
            pd.set_untracked_weights_owner(None)


def _vision_config(config: dict, spec: EncoderSpec) -> dict:
    """What ``check_encoder`` reads, from the config of an encoder loaded
    with its vision tower: the text model's sizes, and the spec's model
    type when the multimodal type is the one the spec names."""
    text = config.get("text_config") or {}
    kind = config.get("model_type")
    return {"model_type": spec.model_type if kind == spec.vision_type
            else kind,
            "hidden_size": text.get("hidden_size"),
            "num_hidden_layers": text.get("num_hidden_layers")}


def _drop_head(model) -> None:
    """``drop_output_head`` on a loaded encoder, with the mapped bytes of
    the matrix taken out of the free-memory reading. Bytes the drop
    returned to MLX were never part of that registration."""
    from gmlx.gen.prefill_decay import deduct_untracked_weights
    from gmlx.load.loader import collect_after_load

    before = mx.get_active_memory()
    held = drop_output_head(model)
    if not held:
        return
    collect_after_load()
    tracked = max(0, before - mx.get_active_memory())
    deduct_untracked_weights(max(0, held - tracked),
                             getattr(model, "_kq_weights_key", None))


def load_encoder(path: str, spec: EncoderSpec, *, override: bool = False,
                 mmproj: str | None = None, log=print) -> EncoderHandle:
    """Load an owned text encoder GGUF through the text loader, without
    its output matrix. The header is checked against the spec before any
    weights load. With ``mmproj`` the encoder loads with its vision tower,
    for a family that conditions on reference images."""
    from gmlx.load.loader import weights_source_key
    from gmlx.load.preflight import find_split_shards

    if mmproj is not None and spec.vision_type is None:
        raise ImageModelError(
            f"the {spec.name} text encoder takes no mmproj: its family "
            "conditions on no reference images")
    check_encoder_file(_header_view(path), spec, override=override)
    if mmproj is not None:
        check_mmproj_file(_header_view(mmproj), spec, override=override)
    model, config, tokenizer, processor, owner = _load_owned(path, mmproj)
    if mmproj is not None:
        config = _vision_config(config, spec)
    check_encoder(model, config, spec, override=True)
    _drop_head(model)
    tower = " with its vision tower" if mmproj is not None else ""
    read = ("read after the final norm" if spec.final_norm else
            "tap after layer " + ", ".join(str(n) for n in spec.tap_layers)
            + f" of {spec.layers}")
    log(f"[image] text encoder {spec.name}{tower}, {read}")
    shards = find_split_shards(path)
    key = (weights_source_key(*shards) if mmproj is None
           else weights_source_key(*shards, mmproj))
    handle = EncoderHandle(model, tokenizer, spec, owned=True,
                           source_key=key, weights_owner=owner,
                           processor=processor)
    if spec.fp32_norms:
        handle.norms = read_norms(path, model, spec)
    return handle


def borrow_encoder(model, tokenizer, config: dict, spec: EncoderSpec, *,
                   override: bool = False,
                   gguf: str | None = None) -> EncoderHandle:
    """Wrap a resident chat model as the encoder. It is checked against the
    spec and never released. ``gguf`` is the chat model's file, which a
    spec with F32 norms reads them from; without it an encode runs on the
    model's own norms."""
    check_encoder(model, config, spec, override=override)
    digests = json.dumps(model_identity(model, spec), sort_keys=True)
    norms = (read_norms(gguf, model, spec) if spec.fp32_norms and gguf
             else {})
    key = "borrowed:" + hashlib.sha256(
        (digests + ("\x00fp32_norms" if norms else "")).encode()).hexdigest()
    handle = EncoderHandle(model, tokenizer, spec, owned=False,
                           source_key=key)
    handle.norms = norms
    return handle
