"""Text encoder tap for image conditioning.

An image family conditions its DiT on the hidden states of a stock chat
model, read after a fixed layer and before the final norm. ``tap`` walks
the loaded model's own layer list and stops there. It changes and
subclasses no stock class.

Each prompt is rendered into the family's fixed template, which the family
owns and never reads from the GGUF chat template, and truncated to the
family's token cap. It is encoded alone at batch 1 with no padding, so the
layers take the causal mask and no attention mask is built. With causal
attention and relative rotary positions, the real tokens of a padded and
masked reference batch take the same values up to rounding.

An ``EncoderHandle`` owns the encoder it loaded and releases it on request,
so the DiT and the VAE can use its memory. A borrowed resident chat model
is never released.
"""

from __future__ import annotations

import hashlib
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

    @property
    def template_id(self) -> str:
        return hashlib.sha256(self.template.encode()).hexdigest()


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
    """The encoder spec of a family, given as a name or a ``Family``."""
    name = family if isinstance(family, str) else family.name
    try:
        return ENCODERS[name]
    except KeyError:
        raise ImageModelError(
            f"image family {name!r} has no text encoder in gmlx") from None


def render(spec: EncoderSpec, prompt: str) -> str:
    """The prompt in the family template."""
    head, sep, tail = spec.template.partition("{}")
    if not sep:
        raise ValueError(f"template of {spec.name} has no prompt slot")
    return head + prompt + tail


def token_ids(spec: EncoderSpec, tokenizer, prompt: str) -> list[int]:
    """Token ids of the rendered prompt, cut at the family's token cap. The
    cap applies to the whole rendered text, as in the reference."""
    ids = list(tokenizer.encode(render(spec, prompt), add_special_tokens=False))
    if spec.max_tokens is not None:
        ids = ids[: spec.max_tokens]
    if not ids:
        raise ValueError("the rendered prompt has no tokens")
    return ids


def _text_stack(model):
    """The embedding and the decoder layer list of a loaded text model."""
    inner = getattr(model, "model", None)
    embed = getattr(inner, "embed_tokens", None)
    layers = getattr(inner, "layers", None)
    if embed is None or layers is None:
        raise TypeError(
            f"{type(model).__name__} has no model.embed_tokens and "
            "model.layers to tap")
    return embed, layers


def tap(model, ids, *, layers: tuple) -> list[mx.array]:
    """Hidden states after running the first ``n`` decoder layers, for each
    ``n`` in ``layers``, with no final norm. ``ids`` is one prompt's token
    ids. Each result is (1, L, hidden) in the model's activation dtype, and
    layers past the last tap never run."""
    embed, blocks = _text_stack(model)
    stops = sorted(set(layers))
    if not stops or stops[0] < 1 or stops[-1] > len(blocks):
        raise ValueError(f"tap layers {layers} outside 1..{len(blocks)}")
    x = mx.array(np.asarray(ids, np.int32).reshape(1, -1))
    h = embed(x)
    mask = "causal" if h.shape[1] > 1 else None
    out: dict[int, mx.array] = {}
    for i in range(stops[-1]):
        h = blocks[i](h, mask, None)
        if i + 1 in stops:
            out[i + 1] = h
    return [out[n] for n in layers]


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
    layer, each (1, L, hidden)."""
    ids: tuple
    hidden: tuple
    template_id: str
    tap_layers: tuple


class EncoderHandle:
    """A text encoder with its tokenizer and spec. An owned handle can be
    released, a borrowed one cannot."""

    def __init__(self, model, tokenizer, spec: EncoderSpec, *, owned: bool,
                 source_key=None):
        self.model = model
        self.tokenizer = tokenizer
        self.spec = spec
        self.owned = owned
        self.source_key = source_key

    @property
    def released(self) -> bool:
        return self.model is None

    def encode(self, prompt: str) -> TextTap:
        if self.model is None:
            raise RuntimeError(
                f"the {self.spec.name} text encoder was released")
        ids = token_ids(self.spec, self.tokenizer, prompt)
        hidden = tap(self.model, ids, layers=self.spec.tap_layers)
        mx.eval(hidden)
        return TextTap(ids=tuple(ids), hidden=tuple(hidden),
                       template_id=self.spec.template_id,
                       tap_layers=tuple(self.spec.tap_layers))

    def release(self) -> bool:
        """Drop an owned encoder and return its memory. Returns False for a
        borrowed or already released encoder, which stays as it is."""
        if not self.owned or self.model is None:
            return False
        self.model = None
        self.tokenizer = None
        from gmlx.load.loader import collect_after_load

        collect_after_load()
        mx.clear_cache()
        return True


def load_encoder(path: str, spec: EncoderSpec, *, override: bool = False,
                 log=print) -> EncoderHandle:
    """Load an owned text encoder GGUF through the text loader. The header
    is checked against the spec before any weights load."""
    from gmlx.load.loader import load_model, weights_source_key
    from gmlx.load.preflight import find_split_shards

    check_encoder_file(_header_view(path), spec, override=override)
    model, config, tokenizer = load_model(path)
    check_encoder(model, config, spec, override=True)
    log(f"[image] text encoder {spec.name}, tap after layer "
        f"{', '.join(str(n) for n in spec.tap_layers)} of {spec.layers}")
    return EncoderHandle(model, tokenizer, spec, owned=True,
                         source_key=weights_source_key(
                             *find_split_shards(path)))


def borrow_encoder(model, tokenizer, config: dict, spec: EncoderSpec, *,
                   override: bool = False) -> EncoderHandle:
    """Wrap a resident chat model as the encoder. It is checked against the
    spec and never released."""
    check_encoder(model, config, spec, override=override)
    return EncoderHandle(model, tokenizer, spec, owned=False)
