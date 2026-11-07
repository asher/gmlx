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

import hashlib
import os
from dataclasses import dataclass, field
from typing import NamedTuple

import mlx.core as mx

from . import loadlog
from .remap import RemapDecision, parse_gguf_name

_A_SUFFIX = ".lora_a"
_B_SUFFIX = ".lora_b"

# Load-time transforms of a base weight that leave its LoRA factors as they
# are. The unsqueeze gives a 1-D gate its (1, in) matrix shape, and the
# factors of that matrix are the ones the GGUF stores.
_FACTOR_PASSTHROUGH = frozenset({"passthrough", "gate_1d_unsqueeze"})
# Transforms an adapter pair can follow: the passthroughs, plus the q/k
# permute, which reorders the rows of ``b``.
_FACTOR_TRANSFORMS = _FACTOR_PASSTHROUGH | {"qk_permute"}


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
    experts: bool = False     # 3-D pair: a (E, rank, in), b (E, out, rank)
    embedding: bool = False   # token_embd pair: a (n_vocab, rank), b (n_embd, rank)


@dataclass
class LoraAdapter:
    alpha: float
    arch: str | None
    modules: dict = field(default_factory=dict)   # module_path -> LoraModule
    path: str | None = None
    sha256: str = ""          # digest of the file bytes this plan loaded
    nbytes: int = 0           # size of that file


class AdapterIdentity(NamedTuple):
    """What a loaded copy holds in one adapter slot."""
    path: str
    sha256: str
    nbytes: int


_LOAD_TRIES = 3

_TOKEN_EMBD = "token_embd.weight"


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


def _lora_scale(alpha: float, rank: int) -> float:
    """llama.cpp's ``get_scale`` at strength 1: alpha 0 means no scaling."""
    return alpha / rank if alpha else 1.0


def build_adapter_plan(meta: dict, arrays: dict,
                       *, base_arch: str | None = None) -> LoraAdapter:
    """Pure core: build a :class:`LoraAdapter` from a decoded GGUF KV dict + tensor
    dict (no file I/O, so it is unit-testable). ``base_arch`` (the base model's
    architecture) is checked against the adapter's and used for the name remap.

    Raises on a non-lora adapter, an arch mismatch, an incomplete a/b pair,
    or a target tensor that doesn't map to a module (never silently dropped -
    a dropped adapter weight would serve a subtly wrong model).

    The scale follows llama.cpp: ``alpha / rank``, and 1 when ``alpha`` is 0
    or missing. A ``token_embd.weight`` pair is the token embedding's: A is
    ``(n_vocab, rank)`` and B ``(n_embd, rank)``, the transpose of a linear
    pair's layout."""
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

    alpha = float(meta.get("adapter.lora.alpha") or 0.0)

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
        if base_name != _TOKEN_EMBD and base_name.endswith(_TOKEN_EMBD):
            # llama.cpp's suffix test sends this pair down its token
            # embedding path, which never reads it.
            raise ValueError(
                f"adapter targets {base_name}. llama.cpp loads this pair but "
                f"never applies it, so gmlx refuses it.")
        dec = parse_gguf_name(remap_arch, base_name)
        if dec.kind != RemapDecision.KIND_MAP:
            raise ValueError(
                f"adapter targets {base_name!r} which doesn't map to a module "
                f"(arch {remap_arch!r}, {dec.kind}: {dec.reason})")
        module_path = dec.hf_name
        if module_path.endswith(".weight"):
            module_path = module_path[: -len(".weight")]
        transform = "passthrough" if dec.transform in _FACTOR_PASSTHROUGH else dec.transform
        if base_name == _TOKEN_EMBD:
            if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] \
                    or transform != "passthrough":
                raise ValueError(
                    f"{_TOKEN_EMBD}: lora_a (A) {tuple(a.shape)} and lora_b (B) "
                    f"{tuple(b.shape)} are not (n_vocab, r) and (n_embd, r)")
            rank = int(a.shape[1])
            modules[module_path] = LoraModule(
                module_path=module_path, a=a, b=b, rank=rank,
                scale=_lora_scale(alpha, rank), embedding=True)
            continue
        experts = a.ndim == 3 or b.ndim == 3
        if experts:
            # Per-expert stack: GGUF ne order reversed gives a (E, rank, in),
            # b (E, out, rank). Rank is positional here; the shared-dim
            # heuristic would pick from {rank, E}.
            if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] \
                    or a.shape[1] != b.shape[2]:
                raise ValueError(
                    f"{module_path}: expert lora pair lora_a {tuple(a.shape)} / "
                    f"lora_b {tuple(b.shape)} is not (E, r, in) / (E, out, r)")
            rank = int(a.shape[1])
        else:
            rank = _infer_rank(module_path, a, b)
        modules[module_path] = LoraModule(
            module_path=module_path, a=a, b=b, rank=rank,
            scale=_lora_scale(alpha, rank), transform=transform, experts=experts)

    if not modules:
        raise ValueError("adapter GGUF contains no lora_a/lora_b tensor pairs")
    return LoraAdapter(alpha=alpha, arch=arch, modules=modules)


def _file_key(path: str) -> tuple:
    st = os.stat(path)
    return (st.st_ino, st.st_size, st.st_mtime_ns)


def load_lora_adapter(adapter_path: str,
                      *, base_arch: str | None = None) -> LoraAdapter:
    """Read a GGUF LoRA adapter from disk and build its apply plan. The adapter's
    a/b tensors are full-precision (F32), so the wire-byte reader returns them as
    plain arrays (no kquant codec).

    The tensors are copied out of the file, so a later rewrite of the file
    never changes what the loaded copy serves. ``sha256`` is the digest of
    the file bytes. A file that changes while it loads is read again, and
    refused after three tries."""
    from .wire import load_gguf_wire_bytes

    for _ in range(_LOAD_TRIES):
        before = _file_key(adapter_path)
        with open(adapter_path, "rb") as fh:
            digest = hashlib.file_digest(fh, "sha256").hexdigest()
        arrays, _kquant_meta, _arch, meta, _shapes = load_gguf_wire_bytes(
            adapter_path, zero_copy=False, expect_quant=False)
        if _file_key(adapter_path) == before:
            break
    else:
        raise ValueError(f"adapter file {adapter_path} changed while it loaded")
    plan = build_adapter_plan(meta, arrays, base_arch=base_arch)
    plan.path = adapter_path
    plan.sha256 = digest
    plan.nbytes = int(before[1])
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
                       *, base_arch: str | None = None, slot: int = 0) -> int:
    """Apply a GGUF LoRA adapter live over a loaded base text model - no merge (the
    base stays K-quant; the adapter rides alongside in full precision). Returns the
    number of modules installed.

    Applies to the *raw* mlx-lm model, whose leaf paths are the HF names the adapter's
    GGUF-base-name remap targets; a base that keeps its text stack under
    ``language_model`` is entered there. Head counts (for the llama-family q/k de-permute of
    the adapter's ``B``) come from ``config`` (a synthesized dict or a config object,
    with a nested ``text_config`` fallback for multimodal-shaped configs); a
    qk_permute target without them raises in the installer. The adapter GGUF's own
    ``general.architecture`` drives the name remap, so a base/adapter mismatch surfaces
    structurally - a target path with no matching module makes
    :func:`modules.install_lora_adapter` raise, never a silent no-op. ``slot`` is
    the adapter's column in the row channel; a second adapter on an already
    adapted model installs alongside the first (see the installer)."""
    plan = load_lora_adapter(adapter_gguf, base_arch=base_arch)
    return install_gguf_adapter(raw_model, config, plan, slot=slot)


def install_gguf_adapter(raw_model, config, plan: LoraAdapter, *, slot: int = 0) -> int:
    """Install an adapter plan that is already loaded, as
    :func:`apply_gguf_adapter` does. Returns the number of modules
    installed."""
    from .modules import install_lora_adapter

    cfg = _config_as_dict(config)
    text_cfg = cfg.get("text_config") or {}
    n_head = cfg.get("num_attention_heads", text_cfg.get("num_attention_heads"))
    n_head_kv = cfg.get("num_key_value_heads",
                        text_cfg.get("num_key_value_heads", n_head))
    return install_lora_adapter(_text_root(raw_model, plan), plan, n_head=n_head,
                                n_head_kv=n_head_kv, slot=slot, config=cfg)


def _text_root(model, plan):
    """The module the plan's paths are relative to. A multimodal-shaped base
    (the Qwen3.5 hybrids among them) keeps its text stack under
    ``language_model`` while the GGUF name remap writes text paths without
    that prefix, so the install descends into it unless the plan already
    names it."""
    inner = getattr(model, "language_model", None)
    if inner is None or any(p.startswith("language_model.") for p in plan.modules):
        return model
    return inner


_EMBD_TIED = ("adapter trains token_embd.weight, but this model uses its token "
              "embedding as the output head. An embedding adapter needs a separate "
              "output head; llama.cpp cannot apply it either.")
_EMBD_FOLDED = ("adapter targets token_embd.weight, which this model stores as a "
                "Hadamard-rotated table. llama.cpp refuses this pair too.")
_EMBD_GRANITE_SWITCH = ("adapter trains token_embd.weight, but Granite Switch embeds "
                        "its substituted ids with a plain lookup that llama.cpp never "
                        "adapts, so gmlx refuses it.")


def embd_shape_message(a_shape, b_shape, n_vocab: int, n_embd: int) -> str:
    """The refusal for a token_embd pair that does not fit its base.
    Shapes in numpy order: A ``(n_vocab, r)``, B ``(n_embd, r)``."""
    return (f"token_embd LoRA pair (A) {tuple(a_shape)} / (B) {tuple(b_shape)} "
            f"does not fit the embedding ({n_vocab}, {n_embd}) (wrong base model?)")


def _hadamard_names(scan) -> list:
    from .hadamard import KEY_VERSION

    if scan.kv.get(KEY_VERSION) is None:
        return []
    key = "prism.hadamard.weight_names"
    if key in scan.skipped:
        from .headerscan import scan_gguf

        scan = scan_gguf(scan.path, include_tensors=False,
                         array_limit=int(scan.skipped[key]) + 1)
    names = scan.kv.get(key)
    return list(names) if isinstance(names, (list, tuple)) else []


def embedding_refusal(adapter_path: str, base_gguf_path: str) -> str | None:
    """Why the base cannot take the adapter's ``token_embd`` pair, from the
    two headers alone, or None. None too for an adapter with no such pair.
    The installer makes the same checks for callers that skip this one."""
    from .headerscan import scan_gguf
    from .preflight import find_split_shards

    adapter = {t.name: t.shape for t in scan_gguf(adapter_path).tensors}
    a_ne = adapter.get(_TOKEN_EMBD + _A_SUFFIX)
    if a_ne is None:
        return None
    b_ne = adapter.get(_TOKEN_EMBD + _B_SUFFIX)
    shards = [scan_gguf(p) for p in find_split_shards(base_gguf_path)]
    tensors = {t.name: t.shape for s in shards for t in s.tensors}
    head = shards[0]
    name = os.path.basename(adapter_path)
    if "output.weight" not in tensors:
        return f"{name}: {_EMBD_TIED}"
    embd = tensors.get(_TOKEN_EMBD)
    if embd is None or len(embd) != 2 or b_ne is None or len(a_ne) != 2 \
            or len(b_ne) != 2 or a_ne[0] != b_ne[0] \
            or a_ne[1] != embd[1] or b_ne[1] != embd[0]:
        n_embd, n_vocab = (embd if embd is not None and len(embd) == 2 else (0, 0))
        return f"{name}: " + embd_shape_message(
            tuple(reversed(a_ne)), tuple(reversed(b_ne or ())), n_vocab, n_embd)
    if head.kv.get("general.architecture") == "graniteswitch":
        return f"{name}: {_EMBD_GRANITE_SWITCH}"
    if _TOKEN_EMBD in _hadamard_names(head):
        return f"{name}: {_EMBD_FOLDED}"
    return None


def adapter_load_refusal(base_gguf_path: str, adapter_paths, env=None, *,
                         stream=None, mtp: bool = False) -> None:
    """Refuse, before any model build, an adapter the base cannot take,
    from the headers alone. Raises ``ValueError`` with the first refusal
    over ``adapter_paths``; returns None when every adapter can load.
    ``env``, ``stream`` and ``mtp`` describe the build, for the checks that
    depend on it. A base that is not a GGUF file has no header to read."""
    base = str(base_gguf_path)
    if not (base.endswith(".gguf") and os.path.isfile(base)):
        return None
    for path in adapter_paths or ():
        if not os.path.isfile(path):
            continue                       # the load reports a missing file
        why = embedding_refusal(str(path), str(base_gguf_path))
        if why is not None:
            raise ValueError(why)
    return None


def base_tensor_names(gguf_path: str) -> list[str]:
    """Every tensor name of a GGUF, across its shards, read from the
    headers alone."""
    from .headerscan import scan_gguf
    from .preflight import find_split_shards

    return [t.name for shard in find_split_shards(gguf_path)
            for t in scan_gguf(shard).tensors]


def adapter_tensor_names(module_paths, *, base_arch: str,
                         base_names) -> tuple[dict, dict]:
    """Resolve in-memory module paths to the base GGUF tensors their LoRA
    pairs are keyed to. The loader's name remap runs over the base's own
    tensor names and is inverted, so a written pair is found again by
    :func:`build_adapter_plan` and by llama.cpp, whatever the architecture.

    Returns ``(names, refused)``: path -> ``(tensor name, transform)`` for
    the paths an adapter can hold, and path -> reason for the others."""
    index: dict[str, list] = {}
    for name in base_names:
        if not name.endswith(".weight"):
            continue
        dec = parse_gguf_name(base_arch, name)
        if dec.kind != RemapDecision.KIND_MAP or not (dec.hf_name or "").endswith(".weight"):
            continue
        index.setdefault(dec.hf_name[: -len(".weight")], []).append((name, dec.transform))
    names: dict = {}
    refused: dict = {}
    for path in module_paths:
        hits = index.get(path, [])
        if not hits:
            refused[path] = "no tensor of the base GGUF loads into it"
        elif len(hits) > 1:
            refused[path] = f"{len(hits)} tensors of the base GGUF load into it"
        elif hits[0][1] not in _FACTOR_TRANSFORMS:
            refused[path] = (f"its base tensor {hits[0][0]} loads through the {hits[0][1]!r} "
                             "transform, which a LoRA pair cannot follow")
        else:
            names[path] = hits[0]
    return names, refused


def refusal_summary(refused: dict, limit: int = 3) -> str:
    """One line naming the first few refused paths and their reasons."""
    items = sorted(refused.items())
    text = "; ".join(f"{p} ({why})" for p, why in items[:limit])
    return text + (f", and {len(items) - limit} more" if len(items) > limit else "")


def _gguf_py_names(module_paths, *, base_arch: str, n_layers: int) -> tuple[dict, dict]:
    """The gguf-py tensor-name map, for callers that give no base names.
    Only architectures gguf-py knows are covered."""
    import gguf

    arch_enum = {v: k for k, v in gguf.MODEL_ARCH_NAMES.items()}.get(base_arch)
    if arch_enum is None:
        raise ValueError(
            f"base arch {base_arch!r} is not a known gguf MODEL_ARCH, pass the base "
            f"GGUF's tensor names to map module names to GGUF tensor names")
    name_map = gguf.get_tensor_name_map(arch_enum, n_layers)
    names: dict = {}
    refused: dict = {}
    for path in module_paths:
        stem = name_map.get_name(path)
        if stem is None:
            refused[path] = f"gguf-py has no tensor name for it on arch {base_arch!r}"
            continue
        dec = parse_gguf_name(base_arch, stem + ".weight")
        if dec.transform not in _FACTOR_TRANSFORMS:
            refused[path] = (f"its base tensor {stem}.weight loads through the "
                             f"{dec.transform!r} transform, which a LoRA pair cannot follow")
            continue
        names[path] = (stem + ".weight", dec.transform)
    return names, refused


def save_lora_adapter(path: str, modules, *, alpha: float, base_arch: str,
                      n_head: int, n_head_kv: int, n_layers: int,
                      base_names=None) -> int:
    """Write a llama.cpp-format GGUF LoRA adapter - the inverse of
    :func:`load_lora_adapter`.

    ``modules`` is an iterable of ``(module_path, a, b)`` where ``module_path`` is
    the in-memory (HF) module path and ``a`` / ``b`` are the LoRA factors in the
    GGUF/PEFT orientation: ``a`` is ``(rank, in)``, ``b`` is ``(out, rank)``, so
    ``delta_W = b @ a`` has the base weight's ``(out, in)`` shape. Q/K ``b`` is
    forward-permuted to the wire layout (the inverse of the load-time
    :func:`transforms.qk_permute_wire`) so a reader's de-permute recovers it.

    ``base_names`` are the base GGUF's tensor names (:func:`base_tensor_names`).
    With them, each module path maps to the base tensor that loads into it
    (:func:`adapter_tensor_names`), on any architecture the loader reads.
    Without them the map is gguf-py's (``gguf.get_tensor_name_map``), which
    covers the architectures gguf-py knows. Every module is resolved before
    anything is written, and one that cannot be held raises ``ValueError``
    naming it. Returns the number of pairs written."""
    import gguf
    import numpy as np

    from .transforms import qk_permute_wire_inverse

    modules = list(modules)
    paths = [m[0] for m in modules]
    if base_names is not None:
        names, refused = adapter_tensor_names(paths, base_arch=base_arch, base_names=base_names)
    else:
        names, refused = _gguf_py_names(paths, base_arch=base_arch, n_layers=n_layers)
    for mod, (tensor, _transform) in list(names.items()):
        if tensor.endswith(_TOKEN_EMBD):
            # The writer stores the linear layout, which an embedding pair
            # does not use.
            refused[mod] = "embedding pairs are not exported"
            del names[mod]
    if refused:
        raise ValueError(f"a GGUF adapter cannot hold {len(refused)} of the "
                         f"{len(paths)} LoRA modules: {refusal_summary(refused)}")

    writer = gguf.GGUFWriter(path, base_arch)
    writer.add_type(gguf.GGUFType.ADAPTER)
    writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, float(alpha))

    n = 0
    for module_path, a, b in modules:
        # convert_lora_to_gguf keys lora_a/b off <name>.weight
        gguf_base, transform = names[module_path]
        if transform == "qk_permute":
            nh = n_head_kv if module_path.endswith("k_proj") else n_head
            b = qk_permute_wire_inverse(b, nh)
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
