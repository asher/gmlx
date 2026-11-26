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

import dataclasses
import hashlib
import os
import re
import struct
from dataclasses import dataclass, field
from typing import NamedTuple

import mlx.core as mx

from . import loadlog
from .gguf_layout import (
    sensenova_loop_block,
    shared_input_parts,
)
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
    # A pair on a merged shared-input tensor: the module paths of its two
    # parts, first rows first. ``b`` holds the rows of both until the
    # install splits it by the modules' own output rows.
    split: tuple[str, str] | None = None


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
            raise ValueError(per_layer_embd_message(base_name))
        parts = shared_input_parts(base_name)
        if parts is not None:
            modules.update(_split_pair(remap_arch, base_name, parts, a, b, alpha))
            continue
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
        rank_one = a.ndim == 1
        if rank_one:
            # ggml drops the trailing unit dim of a rank-1 lora_a, ne [in, 1].
            a = a.reshape(1, -1)
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
        elif rank_one and b.ndim == 2 and b.shape[1] == 1:
            rank = 1
        else:
            rank = _infer_rank(module_path, a, b)
        modules[module_path] = LoraModule(
            module_path=module_path, a=a, b=b, rank=rank,
            scale=_lora_scale(alpha, rank), transform=transform, experts=experts)

    if not modules:
        raise ValueError("adapter GGUF contains no lora_a/lora_b tensor pairs")
    return LoraAdapter(alpha=alpha, arch=arch, modules=modules)


def _split_pair(arch: str, base_name: str, parts, a, b, alpha: float) -> dict:
    """The plan entries of a pair on a merged shared-input tensor: one per
    part, sharing ``a``."""
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise ValueError(
            f"{base_name}: lora_a {tuple(a.shape)} / lora_b {tuple(b.shape)} "
            f"are not (r, in) / (out, r)")
    paths = []
    for part in parts:
        dec = parse_gguf_name(arch, part)
        if dec.kind != RemapDecision.KIND_MAP or dec.transform not in _FACTOR_PASSTHROUGH:
            raise ValueError(
                f"adapter targets {base_name!r}, whose part {part!r} doesn't map "
                f"to a module a LoRA pair can follow (arch {arch!r})")
        paths.append(dec.hf_name.removesuffix(".weight"))
    rank = int(a.shape[0])
    split = (paths[0], paths[1])
    return {p: LoraModule(module_path=p, a=a, b=b, rank=rank,
                          scale=_lora_scale(alpha, rank), split=split)
            for p in paths}


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
                       *, base_arch: str | None = None, slot: int = 0,
                       strength: float = 1.0) -> int:
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
    adapted model installs alongside the first (see the installer).
    ``strength`` multiplies every module's static scale, for callers that
    publish no row vector."""
    plan = load_lora_adapter(adapter_gguf, base_arch=base_arch)
    return install_gguf_adapter(raw_model, config, plan, slot=slot,
                                strength=strength)


def install_gguf_adapter(raw_model, config, plan: LoraAdapter, *, slot: int = 0,
                         strength: float = 1.0) -> int:
    """Install an adapter plan that is already loaded, as
    :func:`apply_gguf_adapter` does. Returns the number of modules
    installed."""
    from .modules import install_lora_adapter

    if strength != 1.0:
        plan = dataclasses.replace(plan, modules={
            k: dataclasses.replace(m, scale=m.scale * float(strength))
            for k, m in plan.modules.items()})
    cfg = _config_as_dict(config)
    text_cfg = cfg.get("text_config") or {}
    if not isinstance(text_cfg, dict):
        text_cfg = _config_as_dict(text_cfg)
    n_head = cfg.get("num_attention_heads", text_cfg.get("num_attention_heads"))
    n_head_kv = cfg.get("num_key_value_heads",
                        text_cfg.get("num_key_value_heads", n_head))
    root = _text_root(raw_model, plan)
    plan = _split_merged(root, _fold_loop(plan, _visit_loop(cfg, text_cfg)))
    return install_lora_adapter(root, plan, n_head=n_head,
                                n_head_kv=n_head_kv, slot=slot, config=cfg)


_LAYER_RE = re.compile(r"^(.*?\blayers\.)(\d+)(\..*)$")


def _visit_loop(*cfgs) -> dict | None:
    """The layer loop of a SenseNova Loop config, whose file names the
    layers after the loop by visit."""
    for c in cfgs:
        if c.get("model_type") == "sensenova_loop" and c.get("loop"):
            return c["loop"]
    return None


def _fold_loop(plan: LoraAdapter, loop: dict | None) -> LoraAdapter:
    """A looped base holds each layer once. Its file, and an adapter keyed
    to it, name the layers after the loop by visit; move those pairs onto
    the physical layers, as the base load does."""
    if not loop:
        return plan
    start, end, count = int(loop["start"]), int(loop["end"]), int(loop["count"])

    def fold(path: str) -> str:
        m = _LAYER_RE.match(path)
        if m is None:
            return path
        try:
            layer = sensenova_loop_block(int(m.group(2)), start, end, count)
        except ValueError:
            raise ValueError(
                f"adapter targets layer {m.group(2)}, a repeated visit of the "
                f"loop [{start}, {end}); its pairs belong to the layer's own "
                f"block") from None
        return f"{m.group(1)}{layer}{m.group(3)}"

    modules = {}
    for path, lm in plan.modules.items():
        split = tuple(fold(p) for p in lm.split) if lm.split else None
        new = fold(path)
        modules[new] = dataclasses.replace(lm, module_path=new, split=split)
    return dataclasses.replace(plan, modules=modules)


def _out_rows(module) -> int | None:
    base = getattr(module, "base", module)
    weight = getattr(base, "weight", None)
    return int(weight.shape[0]) if weight is not None and weight.ndim == 2 else None


def _split_merged(root, plan: LoraAdapter) -> LoraAdapter:
    """Give each part of a merged pair the rows of ``b`` its module
    outputs."""
    if not any(lm.split for lm in plan.modules.values()):
        return plan
    by_path = dict(root.named_modules())
    modules = {}
    for path, lm in plan.modules.items():
        if not lm.split:
            modules[path] = lm
            continue
        first, second = lm.split
        rows = [_out_rows(by_path.get(p)) for p in (first, second)]
        total = int(lm.b.shape[0])
        if None in rows or rows[0] + rows[1] != total:
            raise ValueError(
                f"LoRA pair on {first} and {second}: lora_b has {total} rows, "
                f"the two modules output {rows[0]} and {rows[1]}")
        b = lm.b[:rows[0]] if path == first else lm.b[rows[0]:]
        modules[path] = dataclasses.replace(lm, b=b, split=None)
    return dataclasses.replace(plan, modules=modules)


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


def per_layer_embd_message(base_name: str) -> str:
    """The refusal for a pair on a table whose name ends in
    ``token_embd.weight`` but is not the token embedding."""
    return (f"adapter targets {base_name}. llama.cpp loads this pair but "
            f"never applies it, so gmlx refuses it.")


def _folded_table(scan) -> bool:
    """Whether the base's ``prism.hadamard`` spec stores the token
    embedding rotated. A malformed spec counts as no fold here; the load
    reports it."""
    from .hadamard import KEY_VERSION, HadamardSpecError, parse_hadamard_spec

    if scan.kv.get(KEY_VERSION) is None:
        return False
    sizes = [n for k, n in scan.skipped.items() if k.startswith("prism.hadamard.")]
    if sizes:
        from .headerscan import scan_gguf

        scan = scan_gguf(scan.path, include_tensors=False, array_limit=max(sizes) + 1)
    try:
        spec = parse_hadamard_spec(scan.kv)
    except HadamardSpecError:
        return False
    return spec is not None and _TOKEN_EMBD in spec.inverse_names


def embedding_refusal(adapter_path: str, base_gguf_path: str) -> str | None:
    """Why the base cannot take the adapter's ``token_embd`` pair, from the
    two headers alone, or None. None too for an adapter with no such pair.
    A pair on a per-layer table is refused whatever the base.
    The installer makes the same checks for callers that skip this one."""
    from .headerscan import scan_gguf
    from .preflight import find_split_shards

    adapter = {t.name: t.shape for t in scan_gguf(adapter_path).tensors}
    name = os.path.basename(adapter_path)
    for t in sorted(adapter):
        base_name = t.rsplit(".", 1)[0] if t.endswith((_A_SUFFIX, _B_SUFFIX)) else ""
        if base_name != _TOKEN_EMBD and base_name.endswith(_TOKEN_EMBD):
            return f"{name}: " + per_layer_embd_message(base_name)
    a_ne = adapter.get(_TOKEN_EMBD + _A_SUFFIX)
    if a_ne is None:
        return None
    b_ne = adapter.get(_TOKEN_EMBD + _B_SUFFIX)
    shards = [scan_gguf(p) for p in find_split_shards(base_gguf_path)]
    tensors = {t.name: t.shape for s in shards for t in s.tensors}
    head = shards[0]
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
    if _folded_table(head):
        return f"{name}: {_EMBD_FOLDED}"
    return None


_GATE_UP_EXPS = ("ffn_gate_exps.weight", "ffn_up_exps.weight",
                 "ffn_gate_up_exps.weight")


def expert_gate_up_refusal(adapter_path: str) -> str | None:
    """Why gmlx cannot apply the adapter's expert pairs, from its header,
    or None. Expert adapters apply to the down projection only."""
    from .headerscan import scan_gguf

    for t in scan_gguf(adapter_path).tensors:
        base_name = t.name[: -len(_A_SUFFIX)] if t.name.endswith(_A_SUFFIX) else ""
        if base_name.endswith(_GATE_UP_EXPS):
            return (f"{os.path.basename(adapter_path)}: adapter targets "
                    f"{base_name}. gmlx applies expert adapters to the down "
                    f"projection only, not the gate or up projections.")
    return None


def adapter_resident_bytes(stored_bytes: int) -> int:
    """What an adapter's tensors cost in memory once loaded: the copy out
    of the file, plus at most that again for the per-dtype tables the
    delta computes with."""
    return 2 * int(stored_bytes)


def adapter_load_refusal(base_gguf_path: str, adapter_paths) -> None:
    """Refuse, before any model build, an adapter the base cannot take,
    from the headers alone. Raises ``ValueError`` with the first refusal
    over ``adapter_paths``; returns None when every adapter can load. A
    base that is not a GGUF file has no header to read."""
    base = str(base_gguf_path)
    if not (base.endswith(".gguf") and os.path.isfile(base)):
        return None
    for path in adapter_paths or ():
        if not os.path.isfile(path):
            continue                       # the load reports a missing file
        try:
            why = (embedding_refusal(str(path), str(base_gguf_path))
                   or expert_gate_up_refusal(str(path)))
        except (ValueError, struct.error, IndexError):
            continue                       # and a file that is not a GGUF
        if why is not None:
            raise ValueError(why)
    return None


def base_tensor_names(gguf_path: str) -> dict[str, tuple]:
    """Every tensor of a GGUF across its shards, from the headers alone:
    file name -> ``((name as loaded, output rows), ...)``
    (:func:`gguf_layout.loaded_tensor_names`). Iterating it gives the
    file's names."""
    from .gguf_layout import loaded_tensor_names
    from .headerscan import scan_gguf
    from .preflight import find_split_shards

    shards = [scan_gguf(shard) for shard in find_split_shards(gguf_path)]
    shapes = {t.name: list(t.shape) for shard in shards for t in shard.tensors}
    kv = shards[0].kv if shards else {}
    return loaded_tensor_names(shapes, kv, kv.get("general.architecture"))


class AdapterTarget(NamedTuple):
    """The base tensor a module's LoRA pair is written to. A part of a
    merged tensor has ``part`` and the output rows of every part."""
    tensor: str
    transform: str
    part: int = 0
    rows: tuple = ()


def adapter_tensor_names(module_paths, *, base_arch: str,
                         base_names) -> tuple[dict, dict]:
    """Resolve in-memory module paths to the base GGUF tensors their LoRA
    pairs are keyed to. The loader's name remap runs over the base's
    tensors as loaded (``base_names``, from :func:`base_tensor_names`; a
    plain list of names is taken as loaded under the same names) and is
    inverted, so a written pair is found again by
    :func:`build_adapter_plan` and by llama.cpp, whatever the architecture.

    Returns ``(names, refused)``: path -> :class:`AdapterTarget` for the
    paths an adapter can hold, and path -> reason for the others."""
    loaded = (base_names.items() if isinstance(base_names, dict)
              else ((n, ((n, None),)) for n in base_names))
    index: dict[str, list] = {}
    for name, parts in loaded:
        if not name.endswith(".weight"):
            continue
        rows = tuple(r for _n, r in parts) if len(parts) > 1 else ()
        for k, (part, _rows) in enumerate(parts):
            dec = parse_gguf_name(base_arch, part)
            if dec.kind != RemapDecision.KIND_MAP or not (dec.hf_name or "").endswith(".weight"):
                continue
            index.setdefault(dec.hf_name[: -len(".weight")], []).append(
                AdapterTarget(name, dec.transform, k, rows))
    names: dict = {}
    refused: dict = {}
    for path in module_paths:
        hits = index.get(path, [])
        if not hits:
            refused[path] = "no tensor of the base GGUF loads into it"
        elif len(hits) > 1:
            refused[path] = f"{len(hits)} tensors of the base GGUF load into it"
        elif hits[0].rows and hits[0].transform not in _FACTOR_PASSTHROUGH:
            refused[path] = (f"its base tensor {hits[0].tensor} loads through the "
                             f"{hits[0].transform!r} transform, which a LoRA pair on "
                             "a merged tensor cannot follow")
        elif hits[0].transform not in _FACTOR_TRANSFORMS:
            refused[path] = (f"its base tensor {hits[0].tensor} loads through the {hits[0].transform!r} "
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
        names[path] = AdapterTarget(stem + ".weight", dec.transform)
    return names, refused


def _merge_parts(tensor: str, members) -> tuple:
    """One pair on a merged tensor from the pairs of its parts: ``a``
    stacks the parts' factors and ``b`` places each part's rows in its own
    columns, zero for a part not trained. The adapter's one alpha divides
    by the stacked rank, so ``b`` is scaled by the part count to keep each
    part's ``alpha / rank``."""
    import numpy as np

    rows = members[0][0].rows
    ranks = {int(a.shape[0]) for _t, a, _b in members}
    if len(ranks) != 1:
        raise ValueError(f"the parts of {tensor} were trained at ranks {sorted(ranks)}")
    rank = ranks.pop()
    members = sorted(members, key=lambda m: m[0].part)
    stacked = len(members) * rank
    a = np.concatenate([np.array(a.astype(mx.float32)) for _t, a, _b in members])
    b = np.zeros((sum(rows), stacked), np.float32)
    offsets = np.cumsum((0,) + rows)
    for col, (target, _a, part_b) in enumerate(members):
        b[offsets[target.part]:offsets[target.part + 1], col * rank:(col + 1) * rank] = (
            np.array(part_b.astype(mx.float32)) * len(members))
    return mx.array(a), mx.array(b)


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
    naming it. The parts of a merged tensor (``attn_qkv_gate``) go out as
    one pair on it. Returns the number of modules written."""
    import gguf
    import numpy as np

    from .transforms import qk_permute_wire_inverse

    modules = list(modules)
    paths = [m[0] for m in modules]
    if base_names is not None:
        names, refused = adapter_tensor_names(paths, base_arch=base_arch, base_names=base_names)
    else:
        names, refused = _gguf_py_names(paths, base_arch=base_arch, n_layers=n_layers)
    for mod, target in list(names.items()):
        if target.tensor.endswith(_TOKEN_EMBD):
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

    pairs: dict[str, list] = {}
    for module_path, a, b in modules:
        target = names[module_path]
        if target.transform == "qk_permute":
            nh = n_head_kv if module_path.endswith("k_proj") else n_head
            b = qk_permute_wire_inverse(b, nh)
        pairs.setdefault(target.tensor, []).append((target, a, b))
    n = 0
    for tensor, members in pairs.items():
        a, b = _merge_parts(tensor, members) if members[0][0].rows else members[0][1:]
        # convert_lora_to_gguf keys lora_a/b off <name>.weight
        writer.add_tensor(tensor + ".lora_a", np.array(a.astype(mx.float32)))
        writer.add_tensor(tensor + ".lora_b", np.array(b.astype(mx.float32)))
        n += len(members)

    if n == 0:
        writer.close()
        raise ValueError("no LoRA modules to write")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return n
