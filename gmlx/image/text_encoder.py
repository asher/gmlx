"""Text encoder tap for image conditioning.

An image family conditions its DiT on the hidden states of a chat model
as the text loader builds it, which for Qwen3 is the copied mlx-lm class.
The states are read after a fixed layer and before the final norm. ``tap``
walks the loaded model's own layer list and stops there. It changes and
subclasses no class of the model.

Each prompt is rendered into the family's fixed template, which the family
owns and never reads from the GGUF chat template, and truncated to the
family's token cap. It is encoded alone at batch 1 with no padding, so the
layers take the causal mask and no attention mask is built. With causal
attention and relative rotary positions, the real tokens of a padded and
masked reference batch take the same values up to rounding. A family
whose template opens with a fixed turn drops the rows of that turn from
the conditioning, after they ran: the rows it keeps attended to them.

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
import hashlib
import importlib
import json
import os
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .fingerprint import ImageModelError, canonical_bf16


@dataclass(frozen=True)
class IdentityTensor:
    """A small norm tensor that identifies the encoder weights, by its GGUF
    name, its path in the loaded model and the sha256 of its bf16 bits."""
    gguf_name: str
    model_path: str
    sha256: str


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

    @property
    def template_id(self) -> str:
        """sha256 of what turns a prompt into the rows of a conditioning
        beside the token ids: the template, and the count of dropped rows
        when the spec drops any, and that the features stack the taps when
        they do."""
        text = self.template
        if self.drop:
            text += f"\x00drop={self.drop}"
        if self.stack:
            text += "\x00stack"
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
    """The prompt in the family template. An empty prompt is rendered as
    the spec's ``empty`` text."""
    head, sep, tail = spec.template.partition("{}")
    if not sep:
        raise ValueError(f"template of {spec.name} has no prompt slot")
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


def _text_stack(model):
    """The embedding and the decoder layer list of a loaded text model,
    or of the language model of a multimodal one."""
    inner = getattr(model, "model", None)
    if inner is None:
        inner = getattr(getattr(model, "language_model", None), "model", None)
    embed = getattr(inner, "embed_tokens", None)
    layers = getattr(inner, "layers", None)
    if embed is None or layers is None:
        raise TypeError(
            f"{type(model).__name__} has no model.embed_tokens and "
            "model.layers to tap")
    return embed, layers


def tap(model, ids, *, layers: tuple,
        activation: str | None = None) -> list[mx.array]:
    """Hidden states after running the first ``n`` decoder layers, for each
    ``n`` in ``layers``, with no final norm. ``ids`` is one prompt's token
    ids. Each result is (1, L, hidden) in the model's activation dtype, and
    layers past the last tap never run. With ``activation`` the embeddings
    are cast to that dtype, and the layers run in it by promotion over the
    weights as loaded."""
    embed, blocks = _text_stack(model)
    stops = sorted(set(layers))
    if not stops or stops[0] < 1 or stops[-1] > len(blocks):
        raise ValueError(f"tap layers {layers} outside 1..{len(blocks)}")
    x = mx.array(np.asarray(ids, np.int32).reshape(1, -1))
    h = embed(x)
    if activation is not None:
        h = h.astype(getattr(mx, activation))
    mask = "causal" if h.shape[1] > 1 else None
    out: dict[int, mx.array] = {}
    for i in range(stops[-1]):
        h = blocks[i](h, mask, None)
        if i + 1 in stops:
            out[i + 1] = h
    return [out[n] for n in layers]


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
    obj = model
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part, None)
        if obj is None:
            return None
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
    of every tap side by side in tap order, (L - drop, taps * hidden)."""
    ids: tuple
    hidden: tuple
    template_id: str
    tap_layers: tuple
    drop: int = 0
    stack: bool = False

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
        ids = token_ids(self.spec, self.tokenizer, prompt)
        with vision_rope(self.model, self.spec):
            hidden = tap(self.model, ids, layers=self.spec.tap_layers,
                         activation=self.spec.activation)
            mx.eval(hidden)
        return TextTap(ids=tuple(ids), hidden=tuple(hidden),
                       template_id=self.spec.template_id,
                       tap_layers=tuple(self.spec.tap_layers),
                       drop=self.spec.drop, stack=self.spec.stack)

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
    log(f"[image] text encoder {spec.name}{tower}, tap after layer "
        f"{', '.join(str(n) for n in spec.tap_layers)} of {spec.layers}")
    shards = find_split_shards(path)
    key = (weights_source_key(*shards) if mmproj is None
           else weights_source_key(*shards, mmproj))
    return EncoderHandle(model, tokenizer, spec, owned=True, source_key=key,
                         weights_owner=owner, processor=processor)


def borrow_encoder(model, tokenizer, config: dict, spec: EncoderSpec, *,
                   override: bool = False) -> EncoderHandle:
    """Wrap a resident chat model as the encoder. It is checked against the
    spec and never released."""
    check_encoder(model, config, spec, override=override)
    digests = json.dumps(model_identity(model, spec), sort_keys=True)
    key = "borrowed:" + hashlib.sha256(digests.encode()).hexdigest()
    return EncoderHandle(model, tokenizer, spec, owned=False, source_key=key)
