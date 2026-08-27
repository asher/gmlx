"""LoRA finetuning on a K-quant GGUF base - the creation half of GGUF LoRA.

mlx-lm's own tuner trains the adapter; the mlx-kquant patch makes the frozen GGUF
``KQuantLinear`` leaves adaptable, and gradient flows through the quant matmul via
the extension's ``vjp`` (only the adapter trains, the base carries no float copy /
no optimizer state). The trained adapter is written as a llama.cpp GGUF
(:func:`adapter.save_lora_adapter`) so it round-trips straight back into the
inference / serving path - **GGUF in, GGUF out**, no safetensors, no merge.

Mirrors the ``mlx-kquant lora`` walkthrough, but the base is a GGUF (loaded
in-memory by :func:`loader.load_model`, sidestepping mlx-lm's HF-dir tokenizer /
config assumptions) and the output is a GGUF adapter. LoRA only - mlx-lm's DoRA
dispatch does not consult ``to_lora`` on a kquant base.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from types import SimpleNamespace

from .adapter import save_lora_adapter

_A, _B = ".lora_a", ".lora_b"


def lora_modules_to_gguf(model) -> list:
    """Trained LoRA factors from an mlx-lm-adapted model, in GGUF/PEFT orientation.

    mlx-lm's ``LoRALinear`` stores ``lora_a`` ``(in, rank)`` / ``lora_b``
    ``(rank, out)`` with forward ``z = (x @ lora_a) @ lora_b``; the GGUF format is
    ``a = lora_a.T`` ``(rank, in)`` / ``b = lora_b.T`` ``(out, rank)`` so the
    loader's ``(x @ a.T) @ b.T`` reproduces the delta exactly. The trainable
    parameters are keyed by the in-memory module path (``<path>.lora_a/.lora_b``),
    which is what :func:`adapter.save_lora_adapter` maps to GGUF tensor names.
    """
    from mlx.utils import tree_flatten

    params = dict(tree_flatten(model.trainable_parameters()))
    a_by, b_by = {}, {}
    for key, arr in params.items():
        if key.endswith(_A):
            a_by[key[: -len(_A)]] = arr
        elif key.endswith(_B):
            b_by[key[: -len(_B)]] = arr
    return [(mp, a_by[mp].T, b_by[mp].T) for mp in sorted(set(a_by) & set(b_by))]


def save_trained_adapter(model, config, *, base_arch: str, out_path: str,
                         rank: int, scale: float) -> int:
    """Write a model's trained LoRA layers as a llama.cpp GGUF adapter. ``alpha``
    is recovered as ``scale * rank`` so the loader's ``alpha / rank`` recomputes
    the trained ``scale``."""
    modules = lora_modules_to_gguf(model)
    if not modules:
        raise ValueError("model has no trained LoRA layers to save")
    n_head = config["num_attention_heads"]
    n_head_kv = config.get("num_key_value_heads", n_head)
    n_layers = config["num_hidden_layers"]
    return save_lora_adapter(
        out_path, modules, alpha=float(scale) * int(rank), base_arch=base_arch,
        n_head=n_head, n_head_kv=n_head_kv, n_layers=n_layers)


def train_lora(gguf_path: str, data: str, out_path: str, *, iters: int = 150,
               batch_size: int = 4, num_layers: int = 8, rank: int = 8,
               scale: float = 20.0, dropout: float = 0.0,
               learning_rate: float = 1e-4, max_seq_length: int = 2048,
               val_batches: int = 25, steps_per_report: int = 10,
               steps_per_eval: int = 200, seed: int = 0,
               hf_source: str | None = None) -> tuple[str, int]:
    """Train a LoRA adapter on a GGUF base and write it as a GGUF. Returns
    ``(out_path, n_modules)``. The train loop runs on the GPU."""
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.lora import LoRALinear
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from mlx_kquant.mlx_lm_patch import patch_mlx_lm_lora

    from . import loadlog
    from .loader import load_model
    from .preflight import preflight

    mx.random.seed(seed)
    patch_mlx_lm_lora()  # KQuantLinear.to_lora + rely on the extension's vjp
    base_arch = preflight(gguf_path, hf_source=hf_source).arch
    with loadlog.load_ui(False, gguf_path):
        model, config, tokenizer = load_model(gguf_path, hf_source=hf_source)

    linear_to_lora_layers(model, num_layers,
                          {"rank": rank, "scale": scale, "dropout": dropout})
    model.freeze()
    model.apply_to_modules(
        lambda _k, m: m.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
        if isinstance(m, LoRALinear) else None)

    # Feature keys carry mlx-lm's own string defaults (not None): create_dataset
    # reads them via getattr(config, key, default), so a None here would shadow the
    # default and break format auto-detection (chat / prompt+completion / text).
    ds_args = SimpleNamespace(
        data=data, train=True, test=False, hf_dataset=None,
        chat_feature="messages", prompt_feature="prompt",
        completion_feature="completion", text_feature="text",
        mask_prompt=False)
    train_set, val_set, _ = load_dataset(ds_args, tokenizer)

    model.train()
    opt = optim.Adam(learning_rate=learning_rate)
    # mlx-lm's trainer unconditionally writes a final safetensors to adapter_file
    # (steps_per_save only governs the *periodic* ones) - point it at a scratch dir
    # so the only artifact left on disk is our GGUF, written below.
    with tempfile.TemporaryDirectory() as scratch:
        args = TrainingArgs(
            batch_size=batch_size, iters=iters, val_batches=val_batches,
            steps_per_report=steps_per_report, steps_per_eval=steps_per_eval,
            steps_per_save=iters + 1,  # suppress the periodic safetensors snapshots
            max_seq_length=max_seq_length,
            adapter_file=os.path.join(scratch, "mlx_lm_final.safetensors"))
        train(model, opt, CacheDataset(train_set), CacheDataset(val_set), args=args)

    n = save_trained_adapter(model, config, base_arch=base_arch,
                             out_path=out_path, rank=rank, scale=scale)
    return out_path, n


def cmd_train(argv: list[str], prog: str = "gmlx train") -> int:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Train a LoRA adapter on a K-quant GGUF base and write it as a "
        "GGUF adapter (round-trips into `gmlx serve --adapter`). LoRA only.")
    p.add_argument("model", help="Path to the base GGUF (sharded ok), or a "
                   "server-config model id/alias when --config is set (or a "
                   "default config exists).")
    p.add_argument("--config", default=None, metavar="FILE",
                   help="Server config to resolve the base model name against when "
                        "it isn't a file on disk (default: the first existing "
                        "default config).")
    p.add_argument("--data", required=True, metavar="DIR|ID",
                   help="Dataset directory (train.jsonl/valid.jsonl) or an HF "
                        "dataset id (needs `pip install datasets`).")
    p.add_argument("--adapter-out", required=True, metavar="PATH",
                   help="Output path for the trained .gguf adapter.")
    p.add_argument("--iters", type=int, default=150,
                   help="Training iterations (default 150).")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Examples per training step (default 4).")
    p.add_argument("--num-layers", type=int, default=8, metavar="N",
                   help="Number of top transformer layers to adapt (default 8).")
    p.add_argument("--rank", type=int, default=8,
                   help="LoRA rank (default 8).")
    p.add_argument("--scale", type=float, default=20.0,
                   help="LoRA scale; alpha = scale x rank, recovered on load (default 20.0).")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="LoRA dropout (default 0.0).")
    p.add_argument("--learning-rate", type=float, default=1e-4,
                   help="Adam learning rate (default 1e-4).")
    p.add_argument("--max-seq-length", type=int, default=2048,
                   help="Max training sequence length in tokens (default 2048).")
    p.add_argument("--val-batches", type=int, default=25,
                   help="Validation batches per eval, -1 = full set (default 25).")
    p.add_argument("--steps-per-report", type=int, default=10,
                   help="Train-loss report interval in steps (default 10).")
    p.add_argument("--steps-per-eval", type=int, default=200,
                   help="Validation-loss interval in steps (default 200).")
    p.add_argument("--seed", type=int, default=0,
                   help="PRNG seed (default 0).")
    p.add_argument("--hf-source", default=None, metavar="ID|DIR",
                   help="HF repo id for tokenizer/config fallback "
                        "(rarely needed).")
    a = p.parse_args(argv)

    base = a.model
    # A bare name (no path separator, not a .gguf file on disk) resolves as a
    # server-config model id/alias - same rule as `run`/`chat`.
    if (not os.path.exists(os.path.expanduser(base))
            and "/" not in base and os.sep not in base
            and not base.lower().endswith(".gguf")):
        from . import config as cfgmod
        try:
            cfg, cfg_path = cfgmod.load_cli_config(a.config)
            rm = cfgmod.resolve_cli_model(base, cfg) if cfg is not None else None
        except cfgmod.ConfigError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        if rm is not None:
            print(f"[config] '{base}' -> {rm.path}  (from {cfg_path})")
            base = rm.path

    # Fail on a bad --data before the (long) model load: a path-shaped value
    # that isn't a dataset directory would otherwise surface as a confusing
    # HF repo-id validation error after the whole base model is in memory.
    data_dir = os.path.expanduser(a.data)
    if a.data.startswith((".", "/", "~")) or os.path.exists(data_dir):
        if not os.path.isdir(data_dir):
            print(f"error: --data {a.data}: no such directory (need "
                  f"train.jsonl/valid.jsonl inside, or pass an HF dataset id)",
                  file=sys.stderr)
            return 2
        if not os.path.exists(os.path.join(data_dir, "train.jsonl")):
            print(f"error: --data {a.data}: no train.jsonl inside",
                  file=sys.stderr)
            return 2

    adapter_out = os.path.abspath(os.path.expanduser(a.adapter_out))
    # Prove the output path is writable before training: the GGUF writer only
    # opens it after the run completes, and a bad path there would discard
    # every trained weight.
    try:
        os.makedirs(os.path.dirname(adapter_out) or ".", exist_ok=True)
        probe = adapter_out + ".probe"
        with open(probe, "wb"):
            pass
        os.remove(probe)
    except OSError as e:
        print(f"error: --adapter-out is not writable: {e}", file=sys.stderr)
        return 2

    out, n = train_lora(
        os.path.abspath(os.path.expanduser(base)), a.data,
        adapter_out,
        iters=a.iters, batch_size=a.batch_size, num_layers=a.num_layers,
        rank=a.rank, scale=a.scale, dropout=a.dropout,
        learning_rate=a.learning_rate, max_seq_length=a.max_seq_length,
        val_batches=a.val_batches, steps_per_report=a.steps_per_report,
        steps_per_eval=a.steps_per_eval, seed=a.seed, hf_source=a.hf_source)
    print(f"[gmlx] wrote {n}-module LoRA adapter -> {out}")
    return 0
