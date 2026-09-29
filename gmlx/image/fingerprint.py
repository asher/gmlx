"""Identify an image DiT GGUF from its header.

``identify`` matches tensor names and shapes (GGUF order) against each
family's rules after the family's prefix strip, with the arch string as a
hint. The first family that matches wins. A family can also refuse a file
by name, for a relative that shares its layout. Files no family claims are
checked against a list of other known diffusion layouts, then refused with
a summary of the header.

The variant of a family whose variants share every shape comes from a hash
of a few small tensors, each canonicalized to bf16 bits so the hash does not
depend on whether a converter stored them as F32 or BF16.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    from .registry import Family


class ImageModelError(ValueError):
    """An image GGUF that gmlx refuses, with the reason."""


@dataclass(frozen=True)
class Tensor:
    shape: tuple[int, ...]          # GGUF order, input dim first
    codec: str                      # ggml type name, e.g. "Q4_K", "F32"
    path: str | None = None         # shard holding the bytes
    offset: int = 0                 # absolute file offset
    nbytes: int = 0


@dataclass(frozen=True)
class HeaderView:
    """Header KVs and tensors, with names after any prefix strip."""
    kv: dict
    tensors: dict                   # name -> Tensor, in file order
    prefix: str = ""

    @property
    def arch(self) -> str | None:
        a = self.kv.get("general.architecture")
        return str(a) if a else None

    def has(self, name: str) -> bool:
        return name in self.tensors

    def shape(self, name: str) -> tuple[int, ...] | None:
        t = self.tensors.get(name)
        return None if t is None else t.shape

    def dims(self, name: str) -> tuple[int, ...]:
        """The shape of a tensor the rules already required."""
        t = self.tensors.get(name)
        if t is None:
            raise ImageModelError(f"tensor {name!r} is missing")
        return t.shape

    def count(self, group: str) -> int:
        """Number of distinct indices N over names ``group.N.*``."""
        pat = re.compile(re.escape(group) + r"\.(\d+)\.")
        return len({m.group(1) for n in self.tensors
                    if (m := pat.match(n))})

    def any_suffix(self, suffix: str) -> bool:
        return any(n.endswith(suffix) for n in self.tensors)

    def codecs(self) -> Counter:
        return Counter(t.codec for t in self.tensors.values())


@dataclass(frozen=True)
class Match:
    layout: str = "default"


@dataclass(frozen=True)
class Refusal:
    reason: str


@dataclass(frozen=True)
class Identified:
    family: Family
    config: dict
    prefix: str
    layout: str
    variant: str                    # "unknown" when no identity row matches
    identity_hash: str | None
    view: HeaderView


def view_from_scans(scans: Iterable) -> HeaderView:
    """A HeaderView over the tensors of every shard scan, names unstripped."""
    scans = list(scans)
    tensors: dict[str, Tensor] = {}
    for s in scans:
        for t in s.tensors:
            if t.name in tensors:
                raise ImageModelError(
                    f"tensor {t.name!r} appears in more than one shard")
            tensors[t.name] = Tensor(tuple(t.shape), t.type_name, s.path,
                                     s.data_offset + t.offset, t.nbytes)
    return HeaderView(kv=dict(scans[0].kv), tensors=tensors)


def strip_prefix(view: HeaderView, prefixes: Iterable[str]) -> HeaderView:
    """Strip the first prefix that any name carries.

    Names without the prefix keep their spelling. A stripped name that equals
    an unprefixed name is refused, since one of the two would be lost.
    """
    for p in prefixes:
        if not any(n.startswith(p) for n in view.tensors):
            continue
        out: dict[str, Tensor] = {}
        for n, t in view.tensors.items():
            k = n[len(p):] if n.startswith(p) else n
            if k in out:
                raise ImageModelError(
                    f"prefix strip collision: {p + k!r} and {k!r} both "
                    f"name {k!r} once {p!r} is stripped")
            out[k] = t
        return HeaderView(kv=view.kv, tensors=out, prefix=p)
    return view


# Other diffusion layouts, refused by name. The detection keys follow
# ComfyUI-GGUF's convert.py (Apache-2.0, see licenses/ComfyUI-GGUF-LICENSE).
# A row names a model when every key of one of its key sets is present.
_OTHER_FAMILIES: tuple[tuple[str, tuple[tuple[str, ...], ...]], ...] = (
    ("FLUX", (("double_blocks.0.img_attn.proj.weight",),)),
    ("Stable Diffusion 3", (
        ("transformer_blocks.0.attn.add_q_proj.weight",),
        ("joint_blocks.0.x_block.attn.qkv.weight",))),
    ("AuraFlow", (
        ("double_layers.3.modX.1.weight",),
        ("joint_transformer_blocks.3.ff_context.out_projection.weight",))),
    ("HiDream", ((
        "caption_projection.0.linear.weight",
        "double_stream_blocks.0.block.ff_i.shared_experts.w3.weight"),)),
    ("Cosmos Predict2", ((
        "blocks.0.mlp.layer1.weight",
        "blocks.0.adaln_modulation_cross_attn.1.weight"),)),
    ("HunyuanVideo", ((
        "double_blocks.0.img_attn_proj.weight",
        "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight"),)),
    ("Wan", ((
        "blocks.0.self_attn.norm_q.weight", "text_embedding.2.weight",
        "head.modulation"),)),
    ("LTX-Video", ((
        "adaln_single.emb.timestep_embedder.linear_2.weight",
        "transformer_blocks.27.scale_shift_table",
        "caption_projection.linear_2.weight"),)),
    ("Stable Diffusion XL", (
        ("down_blocks.0.downsamplers.0.conv.weight",
         "add_embedding.linear_1.weight"),
        ("input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight",
         "output_blocks.2.2.conv.weight", "output_blocks.5.2.conv.weight"),
        ("label_emb.0.0.weight",))),
    ("Stable Diffusion 1.x", (
        ("down_blocks.0.downsamplers.0.conv.weight",),
        ("input_blocks.3.0.op.weight", "input_blocks.6.0.op.weight",
         "input_blocks.9.0.op.weight", "output_blocks.2.1.conv.weight",
         "output_blocks.5.2.conv.weight", "output_blocks.8.2.conv.weight"))),
)

_GENERIC_PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def other_family(view: HeaderView) -> str | None:
    for name, key_sets in _OTHER_FAMILIES:
        if any(all(view.has(k) for k in keys) for keys in key_sets):
            return name
    return None


def _text_arch(arch: str | None) -> bool:
    if not arch:
        return False
    from gmlx.load.arch_table import ARCH_TABLE
    from gmlx.load.remap import ARCH_ALIAS

    return arch in ARCH_TABLE or arch in ARCH_ALIAS


def describe_unknown(view: HeaderView) -> str:
    """The refusal text for a file no rule recognizes."""
    arch = view.arch or "none"
    codecs = " ".join(f"{c} x{n}" for c, n in view.codecs().most_common())
    # A numbered group counts its blocks, any other group its tensors.
    blocks: dict[str, set] = {}
    plain: Counter = Counter()
    for n in view.tensors:
        head, _, rest = n.partition(".")
        idx = rest.split(".", 1)[0]
        if idx.isdigit():
            blocks.setdefault(head, set()).add(idx)
        else:
            plain[head] += 1
    heads = dict.fromkeys(n.partition(".")[0] for n in view.tensors)
    group_txt = " ".join(
        f"{h} x{len(blocks[h]) if h in blocks else plain[h]}" for h in heads)
    rank2 = [(n, t.shape) for n, t in view.tensors.items()
             if len(t.shape) == 2][:8]
    shapes = ", ".join(f"{n} {s}" for n, s in rank2)
    return (f"unrecognized image GGUF: arch={arch}, {len(view.tensors)} "
            f"tensors, codecs {codecs}, groups {group_txt}, shapes {shapes}")


def canonical_bf16(raw: bytes, codec: str) -> bytes:
    """bf16 bits of a small float tensor, F32 and F16 rounded to nearest
    even. Raises for any other codec."""
    if codec == "BF16":
        return bytes(raw)
    if codec == "F16":
        raw = np.frombuffer(raw, np.float16).astype(np.float32).tobytes()
    elif codec != "F32":
        raise ImageModelError(f"identity tensor stored as {codec}, "
                              "expected F32, F16 or BF16")
    u = np.frombuffer(raw, np.uint32).astype(np.uint64)
    nan = (u & 0x7FFFFFFF) > 0x7F800000
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    r[nan] = ((u[nan] >> 16) | 0x40).astype(np.uint16)
    return r.tobytes()


def weight_identity(view: HeaderView, names: Iterable[str]) -> str | None:
    """sha256 over the named tensors' bf16 bits, in the order given, or
    None when a tensor is missing, has no file bytes, or is quantized."""
    h = hashlib.sha256()
    for n in names:
        t = view.tensors.get(n)
        if t is None or t.path is None or t.codec not in ("F32", "F16",
                                                         "BF16"):
            return None
        fd = os.open(t.path, os.O_RDONLY)
        try:
            raw = os.pread(fd, t.nbytes, t.offset)
        finally:
            os.close(fd)
        if len(raw) != t.nbytes:
            return None
        h.update(canonical_bf16(raw, t.codec))
    return h.hexdigest()


def _families(name: str | None):
    from .registry import families, get_family

    return (get_family(name),) if name else families()


def identify(view: HeaderView, *, family: str | None = None,
             variant: str | None = None) -> Identified:
    """Classify an unstripped header view. Raises ``ImageModelError``."""
    arch = view.arch
    if _text_arch(arch):
        raise ImageModelError(
            f"arch {arch!r} is a text model, use gmlx.load_model")
    for fam in _families(family):
        v = strip_prefix(view, fam.strip_prefixes)
        res = fam.fingerprint(v)
        if res is None:
            continue
        if isinstance(res, Refusal):
            raise ImageModelError(res.reason)
        if arch not in fam.arch_strings:
            raise ImageModelError(
                f"arch {arch!r} contradicts the tensor layout, which is "
                f"{fam.title} (arch {_arch_list(fam)})")
        return Identified(
            family=fam, config=fam.synth_config(v), prefix=v.prefix,
            layout=res.layout, **_variant(fam, v, variant), view=v)
    v = strip_prefix(view, _GENERIC_PREFIXES)
    if family:
        raise ImageModelError(
            f"not a {_families(family)[0].title} file. {describe_unknown(v)}")
    other = other_family(v)
    if other:
        raise ImageModelError(
            f"this is a {other} GGUF, which gmlx image does not support")
    raise ImageModelError(describe_unknown(v))


def _arch_list(fam) -> str:
    return " or ".join(sorted(repr(a) if a else "none"
                              for a in fam.arch_strings))


def _variant(fam, view: HeaderView, variant: str | None) -> dict:
    if not fam.identity_tensors:
        return {"variant": variant or "default", "identity_hash": None}
    h = weight_identity(view, fam.identity_tensors)
    row = fam.variants.get(h) if h else None
    if variant is None:
        return {"variant": row.name if row else "unknown",
                "identity_hash": h}
    names = fam.variant_names()
    if names and variant not in names:
        raise ImageModelError(
            f"variant {variant!r} is not a {fam.title} variant; known: "
            + ", ".join(names))
    if row is not None and row.name != variant:
        raise ImageModelError(
            f"variant={variant!r} contradicts the weights, which match "
            f"{fam.title} {row.name}")
    return {"variant": variant, "identity_hash": h}
