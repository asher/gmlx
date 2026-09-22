"""Hadamard-folded GGUFs: the ``prism.hadamard`` header and its fold targets.

PrismML's Ternary Bonsai files store most projection weights after a
normalized Sylvester-Walsh-Hadamard rotation of their input dimension,
with one explicit sign vector per input width, and the token embedding
table under the inverse. Every such projection must rotate its input at
run time and the embedding must un-rotate its rows, or the model produces
garbage. This module validates the header the way the reference loader
does (PrismML/llama.cpp, ``src/llama-model.cpp``) and maps the folded
wire names onto module paths; ``hadamard_modules`` applies the rotation.

Forward fold of an input width K: ``y = W_stored @ H_blk(s_K * x)``, the
sign before the transform. ``H_blk`` applies the normalized natural-order
Walsh-Hadamard matrix to each contiguous ``block``-wide chunk of the last
axis and is its own inverse. Embedding rows go the other way:
``h = s_K * H_blk(E_stored[ids])``, the sign after the transform. The
``ssm_out`` input is first permuted from tiled to grouped V-head order
when ``gdn_v_grouped`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Architectures whose every folded projection is a module the loader can
# wrap (hadamard_modules): the fold is refused elsewhere rather than run
# with a stray unrotated matmul.
HADAMARD_ARCHES = frozenset({"qwen35"})

KEY_VERSION = "prism.hadamard.version"
_KEY = "prism.hadamard."
_NS = "prism.hadamard"
_TRANSFORM = "normalized-sylvester-walsh-hadamard"
_AXIS = "input-last-dimension"
_SIGN_MODES = ("identity", "explicit")
_INVERSE_NAME = "token_embd.weight"
# The reference's foldable kinds. The expert stacks are wire tensors gmlx
# maps onto SwitchLinear modules with no rotating counterpart.
_EXPERT_KINDS = frozenset({
    "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps",
    "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
})
_FOLDABLE_KINDS = frozenset({
    "attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate", "attn_output",
    "ffn_gate", "ffn_up", "ffn_down", "ssm_out",
}) | _EXPERT_KINDS


class HadamardSpecError(ValueError):
    """The ``prism.hadamard`` header is malformed or names a fold this
    loader cannot apply."""


@dataclass(frozen=True)
class HadamardSpec:
    block: int
    # width -> int8 vector in {-1, +1}; None in identity mode.
    signs: dict[int, np.ndarray] | None
    weight_names: tuple[str, ...]
    inverse_names: tuple[str, ...]
    gdn_v_grouped: bool


@dataclass(frozen=True)
class FoldTarget:
    """One module's rotation: ``width`` is the input width (the embedding
    dim for the inverse), ``signs`` the width's vector or None, ``perm`` the
    ``(rep, nk, hd)`` tiled-to-grouped permute for ``ssm_out`` or None, and
    ``inverse`` marks the embedding table."""
    width: int
    block: int
    signs: np.ndarray | None
    perm: tuple[int, int, int] | None = None
    inverse: bool = False


def foldable_kind(name: str) -> str | None:
    """The tensor kind of a foldable wire name (``output`` for the head),
    or None when the reference would not fold it."""
    if name == "output.weight":
        return "output"
    if not name.startswith("blk.") or not name.endswith(".weight"):
        return None
    parts = name.split(".")
    if len(parts) != 4 or not parts[1].isdigit():
        return None
    return parts[2] if parts[2] in _FOLDABLE_KINDS else None


def _get(meta: dict, key: str, kind, *, required: bool = True) -> Any:
    v = meta.get(_KEY + key)
    if v is None:
        if required:
            raise HadamardSpecError(f"{_KEY}{key} is missing")
        return None
    if kind is int and isinstance(v, bool):
        raise HadamardSpecError(f"{_KEY}{key} must be an integer, got {v!r}")
    if kind is list:
        if not isinstance(v, (list, tuple)):
            raise HadamardSpecError(f"{_KEY}{key} must be an array")
        return list(v)
    if not isinstance(v, kind):
        raise HadamardSpecError(
            f"{_KEY}{key} must be {kind.__name__}, got {type(v).__name__}")
    return v


def parse_hadamard_spec(meta: dict) -> HadamardSpec | None:
    """Parse and validate the ``prism.hadamard`` keys of a decoded GGUF KV
    dict (the ``meta`` from ``kq.load_gguf``, whose integer arrays are
    complete). Returns None when the file carries no fold. Every check the
    reference loader makes is made here, and a failed one raises
    ``HadamardSpecError``."""
    version = meta.get(KEY_VERSION)
    if version is None:
        return None
    if version != 1:
        raise HadamardSpecError(f"unsupported {KEY_VERSION}: {version}")
    block = _get(meta, "block_size", int)
    transform = _get(meta, "transform", str)
    axis = _get(meta, "axis", str)
    sign_mode = _get(meta, "sign_mode", str)
    weight_names = _get(meta, "weight_names", list)
    if block <= 0 or block & (block - 1):
        raise HadamardSpecError(f"invalid {_KEY}block_size: {block}")
    if transform != _TRANSFORM:
        raise HadamardSpecError(f"unsupported {_KEY}transform: {transform}")
    if axis != _AXIS:
        raise HadamardSpecError(f"unsupported {_KEY}axis: {axis}")
    if sign_mode not in _SIGN_MODES:
        raise HadamardSpecError(f"unsupported {_KEY}sign_mode: {sign_mode}")
    if not weight_names:
        raise HadamardSpecError(f"{_KEY}weight_names is empty")

    signs: dict[int, np.ndarray] | None = None
    if sign_mode == "explicit":
        widths = _get(meta, "sign_widths", list)
        values = _get(meta, "sign_values", list)
        if not widths:
            raise HadamardSpecError(
                f"{_KEY}sign_mode is explicit but sign_widths is empty")
        flat = np.asarray(values, dtype=np.int64).reshape(-1)
        signs = {}
        off = 0
        for width in widths:
            width = int(width)
            if width <= 0 or width % block or off + width > flat.size:
                raise HadamardSpecError(f"invalid {_NS} sign width: {width}")
            vec = flat[off:off + width]
            if not np.all((vec == 1) | (vec == -1)):
                raise HadamardSpecError(f"{_NS} sign values must be +/-1")
            signs[width] = vec.astype(np.int8)
            off += width
        if off != flat.size:
            raise HadamardSpecError(f"{_KEY}sign_values length mismatch")

    seen: set[str] = set()
    for name in weight_names:
        if foldable_kind(name) is None:
            raise HadamardSpecError(
                f"{_NS} weight {name!r} is not on a Hadamard-aware matmul path")
        if name in seen:
            raise HadamardSpecError(f"duplicate {_NS} weight: {name}")
        seen.add(name)

    inverse_names = _get(meta, "inverse_weight_names", list, required=False) or []
    inv_seen: set[str] = set()
    for name in inverse_names:
        if name != _INVERSE_NAME:
            raise HadamardSpecError(
                f"{_NS} weight {name!r} is not an inverse-after-lookup table")
        if name in seen or name in inv_seen:
            raise HadamardSpecError(f"duplicate {_NS} inverse weight: {name}")
        inv_seen.add(name)

    grouped = _get(meta, "gdn_v_grouped", bool, required=False)
    return HadamardSpec(
        block=block,
        signs=signs,
        weight_names=tuple(weight_names),
        inverse_names=tuple(inverse_names),
        gdn_v_grouped=bool(grouped),
    )


def resolve_hadamard_targets(
    spec: HadamardSpec,
    arch: str,
    meta: dict,
    tensor_shapes: dict,
    target_prefix: str = "",
) -> dict[str, FoldTarget]:
    """Map every folded wire name onto its module path (the loader's
    ``parse_gguf_name`` -> ``retarget`` rule, minus ``.weight``) with the
    width, signs and permute that module must apply. Raises
    ``HadamardSpecError`` for an architecture outside ``HADAMARD_ARCHES``
    and ``NotImplementedError`` for a folded expert stack."""
    from .remap import RemapDecision, parse_gguf_name
    from .transforms import retarget

    if arch not in HADAMARD_ARCHES:
        raise HadamardSpecError(
            f"arch {arch!r} is not verified to rotate every Hadamard-folded "
            f"weight (supported: {', '.join(sorted(HADAMARD_ARCHES))})")

    def width_of(name: str) -> int:
        shape = tensor_shapes.get(name)
        if shape is None:
            raise HadamardSpecError(f"{_NS} weight {name!r} is not in the file")
        width = int(shape[0])
        if width % spec.block:
            raise HadamardSpecError(
                f"{_NS} weight {name!r}: input width {width} is not a "
                f"multiple of the block size {spec.block}")
        return width

    def signs_of(width: int) -> np.ndarray | None:
        if spec.signs is None:
            return None
        vec = spec.signs.get(width)
        if vec is None:
            raise HadamardSpecError(
                f"{_NS} has no sign vector for input width {width}")
        return vec

    def path_of(name: str) -> str:
        dec = parse_gguf_name(arch, name)
        if dec.kind != RemapDecision.KIND_MAP or dec.hf_name is None:
            raise HadamardSpecError(
                f"{_NS} weight {name!r} has no module in this model: {dec.reason}")
        if dec.transform != "passthrough":
            raise NotImplementedError(
                f"{_NS} weight {name!r} is loaded through the "
                f"{dec.transform!r} transform, which has no rotating module")
        hf = retarget(dec.hf_name, target_prefix)
        return hf[:-len(".weight")] if hf.endswith(".weight") else hf

    perm = None
    if spec.gdn_v_grouped:
        n_v = meta.get(f"{arch}.ssm.time_step_rank")
        n_k = meta.get(f"{arch}.ssm.group_count")
        if not n_v or not n_k or n_v % n_k:
            raise HadamardSpecError(
                f"{_KEY}gdn_v_grouped needs {arch}.ssm.time_step_rank and "
                f"{arch}.ssm.group_count with the rank a multiple of the "
                f"group count, got {n_v!r} and {n_k!r}")
        perm = (int(n_v), int(n_k))

    targets: dict[str, FoldTarget] = {}
    for name in spec.weight_names:
        kind = foldable_kind(name)
        if kind in _EXPERT_KINDS:
            raise NotImplementedError(
                f"{_NS} weight {name!r} is an expert stack; only per-module "
                f"projections rotate")
        width = width_of(name)
        p = None
        if perm is not None and kind == "ssm_out":
            n_v, n_k = perm
            if width % n_v:
                raise HadamardSpecError(
                    f"{_NS} bad GDN head geometry for {name!r}: width {width} "
                    f"is not a multiple of the V-head count {n_v}")
            p = (n_v // n_k, n_k, width // n_v)
        targets[path_of(name)] = FoldTarget(
            width=width, block=spec.block, signs=signs_of(width), perm=p)
    for name in spec.inverse_names:
        width = width_of(name)
        targets[path_of(name)] = FoldTarget(
            width=width, block=spec.block, signs=signs_of(width), inverse=True)
    return targets


def hadamard_targets_for(
    meta: dict,
    arch: str,
    tensor_shapes: dict,
    target_prefix: str = "",
) -> dict[str, FoldTarget]:
    """The fold targets of a decoded GGUF, or an empty dict for a file with
    no ``prism.hadamard`` header. The loaders call this once per file and
    hand the result to ``hadamard_modules.install_hadamard_modules``."""
    spec = parse_hadamard_spec(meta)
    if spec is None:
        return {}
    return resolve_hadamard_targets(
        spec, arch, meta, tensor_shapes, target_prefix=target_prefix)


def hadamard_matrix(n: int) -> np.ndarray:
    """The normalized natural-order Walsh-Hadamard matrix of size ``n`` (a
    power of two), as float64: ``H[i, j] = (-1)^popcount(i & j) / sqrt(n)``."""
    idx = np.arange(n)
    parity = np.array([bin(v).count("1") & 1 for v in (idx[:, None] & idx[None, :]).ravel()])
    return np.where(parity.reshape(n, n), -1.0, 1.0) / np.sqrt(n)


def permute_grouped(x: np.ndarray, perm: tuple[int, int, int]) -> np.ndarray:
    """Tiled ``[rep, nk, hd]`` to grouped ``[nk, rep, hd]`` feature order
    along the last axis (the reference's ``ggml_permute(x, 0, 2, 1, 3)``)."""
    rep, nk, hd = perm
    lead = x.shape[:-1]
    return x.reshape(*lead, rep, nk, hd).swapaxes(-3, -2).reshape(*lead, rep * nk * hd)


def reference_rotate(x: np.ndarray, target: FoldTarget) -> np.ndarray:
    """The forward fold in float64 numpy (permute, sign, transform), the
    oracle the MLX forms are tested against."""
    xf = np.asarray(x, dtype=np.float64)
    if target.perm is not None:
        xf = permute_grouped(xf, target.perm)
    if target.signs is not None:
        xf = xf * target.signs.astype(np.float64)
    h = hadamard_matrix(target.block)
    lead = xf.shape[:-1]
    chunks = xf.reshape(*lead, -1, target.block) @ h
    return chunks.reshape(*lead, -1)


def reference_rotate_inverse(rows: np.ndarray, target: FoldTarget) -> np.ndarray:
    """The embedding un-rotation in float64 numpy (transform, then sign)."""
    xf = np.asarray(rows, dtype=np.float64)
    h = hadamard_matrix(target.block)
    lead = xf.shape[:-1]
    out = (xf.reshape(*lead, -1, target.block) @ h).reshape(*lead, -1)
    if target.signs is not None:
        out = out * target.signs.astype(np.float64)
    return out
