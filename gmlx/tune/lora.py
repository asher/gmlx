"""LoRA student setup and adapter export, shared by ``gmlx train`` and the
distillation trainer.

The base stays a frozen K-quant GGUF: mlx-kquant's ``to_lora`` patch makes
``KQuantLinear`` leaves adaptable and the extension's matmul ``vjp``
carries the gradient to the adapter, so the base holds no float copy and no
optimizer state. Trained factors are written as a llama.cpp GGUF adapter
through ``gmlx.load.adapter.save_lora_adapter`` and load back into the
serving path with ``--adapter``.
"""
from __future__ import annotations

import os
from collections.abc import Iterable

from gmlx.load.adapter import adapter_tensor_names, save_lora_adapter

_A, _B = ".lora_a", ".lora_b"

# The dense projections of a decoder layer, the target set the distillation
# trainer adapts on every layer. ``gmlx train`` keeps mlx-lm's per-architecture
# defaults instead (``keys=None``).
LORA_KEYS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def lora_scale(rank: int, scale: float | None = None, alpha: float | None = None) -> float:
    """The multiplier on the low-rank product: ``alpha / rank`` when alpha is
    given (the PEFT convention, so the update norm does not grow with the
    rank), else ``scale`` as is (mlx-lm's convention; the GGUF adapter stores
    ``alpha = scale * rank``). Exactly one of the two must be given."""
    if rank < 1:
        raise ValueError(f"LoRA rank must be at least 1, got {rank}")
    if (alpha is None) == (scale is None):
        raise ValueError("give exactly one of scale and alpha")
    if alpha is not None:
        return float(alpha) / rank
    assert scale is not None
    return float(scale)


def _layer_list(model):
    from gmlx.tune.checkpoint import layer_list
    return layer_list(model)


def prepare_lora_student(model, *, rank: int = 8, scale: float | None = 20.0,
                         dropout: float = 0.0, num_layers: int | None = None,
                         keys: Iterable[str] | None = None,
                         alpha: float | None = None) -> int:
    """Adapt ``model`` in place: LoRA on the top ``num_layers`` decoder
    layers (every layer when None) at the module paths in ``keys``
    (mlx-lm's per-architecture defaults when None), the base frozen and the
    adapter factors trainable. Works on a gmlx K-quant base and on an MLX
    float checkpoint. An expert stack (a switch layer) named by the keys
    is left as it was: a GGUF adapter holds matrices only, and the fused
    expert paths stay in force. A row-fused projection pair is split back
    into its members first, since the forward would call the fused module
    in place of an adapted member. Returns the number of adapted modules."""
    from mlx_lm.tuner.lora import LoRALinear, LoRASwitchLinear
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from mlx_kquant.mlx_lm_patch import patch_mlx_lm_lora

    from gmlx.load.modules import drop_fused_children

    multiplier = lora_scale(rank, scale, alpha)
    patch_mlx_lm_lora()   # KQuantLinear.to_lora; idempotent
    drop_fused_children(model)
    n = len(_layer_list(model)) if num_layers is None else num_layers
    config = {"rank": rank, "scale": multiplier, "dropout": dropout}
    if keys is not None:
        config["keys"] = set(keys)
    linear_to_lora_layers(model, n, config)
    for path, m in list(model.named_modules()):
        if isinstance(m, LoRASwitchLinear):
            _set_module(model, path, m.linear)
    model.freeze()
    count = 0

    def unfreeze(_k, m):
        nonlocal count
        if isinstance(m, LoRALinear):
            m.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
            count += 1

    model.apply_to_modules(unfreeze)
    return count


def _set_module(root, path: str, module) -> None:
    """Replace the module at a dotted ``path`` under ``root`` (list
    entries by index)."""
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if isinstance(parent, list) else getattr(parent, part)
    last = parts[-1]
    if isinstance(parent, list):
        parent[int(last)] = module
    else:
        setattr(parent, last, module)


def _selected(path: str, keys: Iterable[str] | None) -> bool:
    if keys is None:
        return True
    return any(path == k or path.endswith("." + k) for k in keys)


def lora_modules_to_gguf(model, keys: Iterable[str] | None = None) -> list:
    """Trained LoRA factors from an mlx-lm-adapted model, in GGUF/PEFT
    orientation, as ``(module_path, a, b)`` sorted by path.

    mlx-lm's ``LoRALinear`` stores ``lora_a`` ``(in, rank)`` and ``lora_b``
    ``(rank, out)`` with forward ``z = (x @ lora_a) @ lora_b``; the GGUF
    format is ``a = lora_a.T`` ``(rank, in)`` and ``b = lora_b.T``
    ``(out, rank)``, so the loader's ``(x @ a.T) @ b.T`` reproduces the
    delta exactly. ``keys`` keeps only the module paths ending in one of
    the given dotted suffixes, which is how a MoE student's expert stacks
    stay out of an adapter meant for its dense projections."""
    from mlx.utils import tree_flatten

    params = dict(tree_flatten(model.trainable_parameters()))
    a_by, b_by = {}, {}
    for key, arr in params.items():
        if key.endswith(_A):
            a_by[key[: -len(_A)]] = arr
        elif key.endswith(_B):
            b_by[key[: -len(_B)]] = arr
    paths = sorted(mp for mp in set(a_by) & set(b_by) if _selected(mp, keys))
    for mp in paths:
        if a_by[mp].ndim != 2 or b_by[mp].ndim != 2:
            raise ValueError(f"the LoRA factors of {mp} have {a_by[mp].ndim} and {b_by[mp].ndim} axes, a GGUF "
                             "adapter holds matrices only (stacked expert factors cannot be exported)")
    return [(mp, a_by[mp].T, b_by[mp].T) for mp in paths]


def adapter_refusals(model, *, base_arch: str, base_names,
                     keys: Iterable[str] | None = None) -> dict:
    """The adapted modules :func:`save_trained_adapter` could not write,
    as path -> reason, empty when the export will succeed. The trainers
    call it before the first step, so a run whose adapter cannot be written
    stops before it trains."""
    paths = [mp for mp, _a, _b in lora_modules_to_gguf(getattr(model, "language_model", model), keys)]
    return adapter_tensor_names(paths, base_arch=base_arch, base_names=base_names)[1]


def save_trained_adapter(model, config, *, base_arch: str, out_path: str,
                         scale: float, rank: int | None = None,
                         keys: Iterable[str] | None = None,
                         base_names=None) -> int:
    """Write a model's trained LoRA layers as a llama.cpp GGUF adapter.
    ``alpha`` is stored as ``scale * rank`` so the loader's ``alpha / rank``
    recomputes the trained ``scale``; the rank is the factors' own, and a
    ``rank`` given that differs from it is refused. A multimodal wrapper
    is read at its ``language_model``, so the module paths match the text
    base the adapter loads onto. ``base_names``, the base GGUF's tensor
    names, key each pair to the tensor that loads into its module, which
    every architecture the loader reads needs and gguf-py covers only for
    its own. Returns the module count."""
    modules = lora_modules_to_gguf(getattr(model, "language_model", model), keys)
    if not modules:
        raise ValueError("model has no trained LoRA layers to save")
    trained = int(modules[0][1].shape[0])
    if rank is not None and int(rank) != trained:
        raise ValueError(f"rank {rank} differs from the trained factors' rank {trained}")
    rank = trained
    cfg = config.get("text_config", config)
    n_head = cfg["num_attention_heads"]
    n_head_kv = cfg.get("num_key_value_heads", n_head)
    n_layers = cfg["num_hidden_layers"]
    # written beside the final name and moved into place, so a failure
    # mid-write leaves no adapter a loader could read
    tmp = out_path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    try:
        n = save_lora_adapter(
            tmp, modules, alpha=float(scale) * int(rank), base_arch=base_arch,
            n_head=n_head, n_head_kv=n_head_kv, n_layers=n_layers, base_names=base_names)
        fd = os.open(tmp, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, out_path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return n


def resolve_model_arg(base: str, config: str | None = None) -> tuple[str, str | None, str | None]:
    """Resolve a ``model`` argument the way ``run`` and ``chat`` do: a bare
    name that is not a path on disk and does not end in ``.gguf`` is a
    server-config model id or alias. Returns ``(path, note, error)``: the
    path to use, a note about the resolution for the log (or None), and an
    error message when the config could not be read (or None)."""
    if (os.path.exists(os.path.expanduser(base))
            or "/" in base or os.sep in base
            or base.lower().endswith(".gguf")):
        return base, None, None
    import gmlx.config as cfgmod
    try:
        cfg, cfg_path = cfgmod.load_cli_config(config)
        rm = cfgmod.resolve_cli_model(base, cfg) if cfg is not None else None
    except cfgmod.ConfigError as e:
        return base, None, str(e)
    if rm is None:
        return base, None, None
    return rm.path, f"'{base}' -> {rm.path}  (from {cfg_path})", None


def probe_writable(path: str) -> str | None:
    """Prove a file can be written at ``path``. Returns the OS error
    message, or None when writable. The folders the probe makes on the way
    are removed again, so a run refused later leaves none behind, and the
    writer makes them when it writes. A path that names a directory is
    refused, since the file's own write would fail only after the run."""
    if os.path.isdir(path) or path.endswith(os.sep):
        return "it is a directory"
    made = []
    d = os.path.abspath(os.path.dirname(path) or ".")
    while not os.path.exists(d) and os.path.dirname(d) != d:
        made.append(d)
        d = os.path.dirname(d)
    err = None
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        probe = path + ".probe"
        with open(probe, "wb"):
            pass
        os.remove(probe)
    except OSError as e:
        err = str(e)
    for d in made:
        try:
            os.rmdir(d)
        except OSError:
            break
    return err
