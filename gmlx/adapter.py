"""Load a llama.cpp GGUF LoRA adapter into a per-module apply plan.

Targets the de-facto-standard format emitted by ``convert_lora_to_gguf.py``: KV
``adapter.type = "lora"`` + ``adapter.lora.alpha``, and for each targeted base
weight ``<base>.weight`` a pair of tensors ``<base>.weight.lora_a`` /
``<base>.weight.lora_b`` keyed to the **base GGUF tensor name**. Adapters are
small and stored in full precision (F32 by default), so the tensors load as plain
arrays through the same C++ reader the base model uses.

This module does the read + validation + a/b pairing + rank/scale + name remap
(GGUF base name -> in-memory HF module path, via :func:`remap.parse_gguf_name`).
The forward wrap lives in :class:`modules.LoRAKQuantLinear`; applying a plan to a
model is :func:`modules.install_lora_adapter`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from . import loadlog
from .remap import RemapDecision, parse_gguf_name

_A_SUFFIX = ".lora_a"
_B_SUFFIX = ".lora_b"


@dataclass
class LoraModule:
    """One targeted Linear's adapter: ``base(x) + scale * (x @ a) @ b``. ``a``/``b``
    are kept in the GGUF's stored orientation; the wrap pins the exact layout."""

    module_path: str          # in-memory module path, e.g. model.layers.0.self_attn.q_proj
    a: mx.array               # lora_a
    b: mx.array               # lora_b
    rank: int
    scale: float              # alpha / rank (PEFT scaling)
    transform: str = "passthrough"   # remap hint: passthrough | qk_permute | ...


@dataclass
class LoraAdapter:
    alpha: float
    arch: str | None
    modules: dict = field(default_factory=dict)   # module_path -> LoraModule
    path: str | None = None


def _infer_rank(module_path: str, a: mx.array, b: mx.array) -> int:
    """The LoRA rank is the dimension ``lora_a`` and ``lora_b`` share (``in->r``,
    ``r->out``) - the small common one. Raise if they share none (malformed pair)."""
    shared = set(a.shape) & set(b.shape)
    if not shared:
        raise ValueError(
            f"{module_path}: lora_a {tuple(a.shape)} and lora_b {tuple(b.shape)} "
            f"share no dimension - cannot infer rank")
    if len(shared) > 1:
        loadlog.verbose_print(
            f"[adapter] {module_path}: lora_a {tuple(a.shape)} / lora_b "
            f"{tuple(b.shape)} share dims {sorted(shared)}; taking rank="
            f"{min(shared)}")
    return min(shared)


def build_adapter_plan(meta: dict, arrays: dict,
                       *, base_arch: str | None = None) -> LoraAdapter:
    """Pure core: build a :class:`LoraAdapter` from a decoded GGUF KV dict + tensor
    dict (no file I/O, so it is unit-testable). ``base_arch`` (the base model's
    architecture) is checked against the adapter's and used for the name remap.

    Raises on a non-lora adapter, a missing ``alpha``, an arch mismatch, an
    incomplete a/b pair, or a target tensor that doesn't map to a module (never
    silently dropped - a dropped adapter weight would serve a subtly wrong model)."""
    a_type = meta.get("adapter.type")
    if a_type != "lora":
        raise ValueError(
            f"not a LoRA adapter GGUF: adapter.type={a_type!r} (expected 'lora')")

    arch = meta.get("general.architecture")
    if base_arch is not None and arch is not None and arch != base_arch:
        raise ValueError(
            f"adapter arch {arch!r} != base arch {base_arch!r} - adapter is for a "
            f"different base model")
    remap_arch = base_arch or arch
    if remap_arch is None:
        raise ValueError("no architecture for the adapter (neither the GGUF's "
                         "general.architecture nor a base_arch was given)")

    alpha = meta.get("adapter.lora.alpha")
    if alpha is None:
        raise ValueError("adapter GGUF has no adapter.lora.alpha - cannot scale")
    alpha = float(alpha)

    # Pair lora_a / lora_b by their shared base tensor name.
    a_by_base: dict = {}
    b_by_base: dict = {}
    for name, arr in arrays.items():
        if name.endswith(_A_SUFFIX):
            a_by_base[name[: -len(_A_SUFFIX)]] = arr
        elif name.endswith(_B_SUFFIX):
            b_by_base[name[: -len(_B_SUFFIX)]] = arr

    modules: dict = {}
    for base_name in sorted(set(a_by_base) | set(b_by_base)):
        a = a_by_base.get(base_name)
        b = b_by_base.get(base_name)
        if a is None or b is None:
            missing = _A_SUFFIX if a is None else _B_SUFFIX
            raise ValueError(
                f"adapter tensor {base_name!r} is missing its {missing} half")
        dec = parse_gguf_name(remap_arch, base_name)
        if dec.kind != RemapDecision.KIND_MAP:
            raise ValueError(
                f"adapter targets {base_name!r} which doesn't map to a module "
                f"(arch {remap_arch!r}, {dec.kind}: {dec.reason})")
        module_path = dec.hf_name
        if module_path.endswith(".weight"):
            module_path = module_path[: -len(".weight")]
        rank = _infer_rank(module_path, a, b)
        modules[module_path] = LoraModule(
            module_path=module_path, a=a, b=b, rank=rank,
            scale=alpha / rank, transform=dec.transform)

    if not modules:
        raise ValueError("adapter GGUF contains no lora_a/lora_b tensor pairs")
    return LoraAdapter(alpha=alpha, arch=arch, modules=modules)


def load_lora_adapter(adapter_path: str,
                      *, base_arch: str | None = None) -> LoraAdapter:
    """Read a GGUF LoRA adapter from disk and build its apply plan. The adapter's
    a/b tensors are full-precision (F32), so the wire-byte reader returns them as
    plain arrays (no kquant codec)."""
    from .loader import load_gguf_wire_bytes

    arrays, _kquant_meta, _arch, meta, _shapes = load_gguf_wire_bytes(
        adapter_path, expect_quant=False)
    plan = build_adapter_plan(meta, arrays, base_arch=base_arch)
    plan.path = adapter_path
    return plan


def _config_as_dict(config) -> dict:
    """A config (synthesized dict, or an mlx-lm/-vlm config object) as a plain dict."""
    if isinstance(config, dict):
        return config
    for attr in ("to_dict", "__dict__"):
        value = getattr(config, attr, None)
        if callable(value):
            return dict(value())
        if isinstance(value, dict):
            return dict(value)
    return {}


def apply_gguf_adapter(raw_model, config, adapter_gguf: str,
                       *, base_arch: str | None = None) -> int:
    """Apply a GGUF LoRA adapter live over a loaded base text model - no merge (the
    base stays K-quant; the adapter rides alongside in full precision). Returns the
    number of modules installed.

    Applies to the *raw* mlx-lm model, whose leaf paths are the HF names the adapter's
    GGUF-base-name remap targets. Head counts (for the llama-family q/k de-permute of
    the adapter's ``B``) come from ``config`` (a synthesized dict or a config object,
    with a nested ``text_config`` fallback for multimodal-shaped configs); a
    qk_permute target without them raises in the installer. The adapter GGUF's own
    ``general.architecture`` drives the name remap, so a base/adapter mismatch surfaces
    structurally - a target path with no matching module makes
    :func:`modules.install_lora_adapter` raise, never a silent no-op."""
    from .modules import install_lora_adapter

    cfg = _config_as_dict(config)
    text_cfg = cfg.get("text_config") or {}
    n_head = cfg.get("num_attention_heads", text_cfg.get("num_attention_heads"))
    n_head_kv = cfg.get("num_key_value_heads",
                        text_cfg.get("num_key_value_heads", n_head))
    plan = load_lora_adapter(adapter_gguf, base_arch=base_arch)
    return install_lora_adapter(raw_model, plan, n_head=n_head, n_head_kv=n_head_kv)


def save_lora_adapter(path: str, modules, *, alpha: float, base_arch: str,
                      n_head: int, n_head_kv: int, n_layers: int) -> int:
    """Write a llama.cpp-format GGUF LoRA adapter - the inverse of
    :func:`load_lora_adapter`.

    ``modules`` is an iterable of ``(module_path, a, b)`` where ``module_path`` is
    the in-memory (HF) module path and ``a`` / ``b`` are the LoRA factors in the
    GGUF/PEFT orientation: ``a`` is ``(rank, in)``, ``b`` is ``(out, rank)``, so
    ``delta_W = b @ a`` has the base weight's ``(out, in)`` shape. Q/K ``b`` is
    forward-permuted to the wire layout (the inverse of the load-time
    :func:`transforms.qk_permute_wire`) so a reader's de-permute recovers it.

    The module-path -> GGUF-tensor-name map is the canonical llama.cpp one
    (``gguf.get_tensor_name_map``), and the transform decision mirrors
    :func:`remap.parse_gguf_name`, so the emitted names + Q/K layout are exactly
    what this module's loader reads back. Returns the number of pairs written."""
    import gguf
    import numpy as np

    from .transforms import qk_permute_wire_inverse

    arch_enum = {v: k for k, v in gguf.MODEL_ARCH_NAMES.items()}.get(base_arch)
    if arch_enum is None:
        raise ValueError(
            f"base arch {base_arch!r} is not a known gguf MODEL_ARCH - cannot map "
            f"module names to GGUF tensor names")
    name_map = gguf.get_tensor_name_map(arch_enum, n_layers)

    writer = gguf.GGUFWriter(path, base_arch)
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, float(alpha))

    n = 0
    for module_path, a, b in modules:
        stem = name_map.get_name(module_path)
        if stem is None:
            raise ValueError(
                f"{module_path!r} has no GGUF tensor name for arch {base_arch!r}")
        gguf_base = stem + ".weight"   # convert_lora_to_gguf keys lora_a/b off <name>.weight
        dec = parse_gguf_name(base_arch, gguf_base)
        if dec.transform == "qk_permute":
            nh = n_head_kv if module_path.endswith("k_proj") else n_head
            b = qk_permute_wire_inverse(b, nh)
        elif dec.transform != "passthrough":
            raise NotImplementedError(
                f"{module_path}: LoRA save for the {dec.transform!r} transform is "
                f"not supported yet")
        writer.add_tensor(gguf_base + ".lora_a", np.array(a.astype(mx.float32)))
        writer.add_tensor(gguf_base + ".lora_b", np.array(b.astype(mx.float32)))
        n += 1

    if n == 0:
        writer.close()
        raise ValueError("no LoRA modules to write")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return n
