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

from gmlx.tune.lora import (  # noqa: F401  (re-exported for callers and tests)
    adapter_refusals,
    lora_modules_to_gguf,
    prepare_lora_student,
    probe_writable,
    resolve_model_arg,
    save_trained_adapter,
)


class TrainRefused(ValueError):
    """A setting the loaded model cannot train with, found after the load."""


def train_lora(gguf_path: str, data: str, out_path: str, *, iters: int = 150,
               batch_size: int = 4, num_layers: int = 8, rank: int = 8,
               scale: float = 20.0, dropout: float = 0.0,
               learning_rate: float = 1e-4, max_seq_length: int = 2048,
               val_batches: int = 25, steps_per_report: int = 10,
               steps_per_eval: int = 200, seed: int = 0,
               hf_source: str | None = None,
               grad_checkpoint: bool = False) -> tuple[str, int]:
    """Train a LoRA adapter on a GGUF base and write it as a GGUF. Returns
    ``(out_path, n_modules)``. The train loop runs on the GPU. Raises
    TrainRefused when the loaded model cannot take a requested setting."""
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.trainer import TrainingArgs, train

    from mlx_kquant.mlx_lm_patch import patch_mlx_lm_lora

    import gmlx.load.loadlog as loadlog
    from gmlx.load.adapter import base_tensor_names, refusal_summary
    from gmlx.load.loader import load_model
    from gmlx.load.preflight import preflight
    from gmlx.tune.attention import install_training_attention
    from gmlx.tune.checkpoint import checkpoint_layers
    from gmlx.tune.gdn import install_training_gdn
    from gmlx.tune.indices import install_index_stop_gradient
    from gmlx.tune.kernels import install_training_switch_gemm

    mx.random.seed(seed)
    patch_mlx_lm_lora()  # KQuantLinear.to_lora + rely on the extension's vjp
    base_arch = preflight(gguf_path, hf_source=hf_source).arch
    base_names = base_tensor_names(gguf_path)
    with loadlog.load_ui(False, gguf_path):
        model, config, tokenizer = load_model(gguf_path, hf_source=hf_source)

    prepare_lora_student(model, rank=rank, scale=scale, dropout=dropout,
                         num_layers=num_layers)
    refused = adapter_refusals(model, base_arch=base_arch, base_names=base_names)
    if refused:
        raise TrainRefused(f"a GGUF adapter cannot hold {len(refused)} of the adapted modules, "
                           f"nothing was trained: {refusal_summary(refused)}")

    # Feature keys carry mlx-lm's own string defaults (not None): create_dataset
    # reads them via getattr(config, key, default), so a None here would shadow the
    # default and break format auto-detection (chat / prompt+completion / text).
    ds_args = SimpleNamespace(
        data=data, train=True, test=False, hf_dataset=None,
        chat_feature="messages", prompt_feature="prompt",
        completion_feature="completion", text_feature="text",
        mask_prompt=False)
    # mlx-lm types the tokenizer as PreTrainedTokenizer; its wrapper is what
    # load_dataset reads at runtime
    train_set, val_set, _ = load_dataset(ds_args, tokenizer)  # pyright: ignore[reportArgumentType]

    model.train()
    opt = optim.Adam(learning_rate=learning_rate)
    # mlx-lm's trainer unconditionally writes a final safetensors to adapter_file
    # (steps_per_save only governs the *periodic* ones) - point it at a scratch dir
    # so the only artifact left on disk is our GGUF, written below.
    restore_attention = install_training_attention(model)
    install_training_gdn(model)   # mlx-lm gated delta layers: the checkpointed scan under training
    if grad_checkpoint:
        # every decoder-layer class, under language_model too; mlx-lm's own
        # grad_checkpoint wraps the class of model.layers[0] alone
        try:
            n_ck = checkpoint_layers(model)
        except ValueError as e:
            restore_attention()
            raise TrainRefused(f"--grad-checkpoint: {e}") from None
        print(f"[train] per-layer checkpointing on {n_ck} layer classes")
    # a router gathers its weights at ids it picked from trained scores,
    # and MLX has no backward for a gather at ids that carry a gradient
    restore_ids = install_index_stop_gradient()
    restore_gemm = install_training_switch_gemm()
    try:
        with tempfile.TemporaryDirectory() as scratch:
            args = TrainingArgs(
                batch_size=batch_size, iters=iters, val_batches=val_batches,
                steps_per_report=steps_per_report, steps_per_eval=steps_per_eval,
                steps_per_save=iters + 1,  # suppress the periodic safetensors snapshots
                max_seq_length=max_seq_length, grad_checkpoint=False,
                adapter_file=os.path.join(scratch, "mlx_lm_final.safetensors"))
            train(model, opt, CacheDataset(train_set), CacheDataset(val_set), args=args)
    finally:
        restore_gemm()
        restore_ids()
        restore_attention()

    n = save_trained_adapter(model, config, base_arch=base_arch,
                             out_path=out_path, rank=rank, scale=scale,
                             base_names=base_names)
    return out_path, n


def _rank(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"an integer is required, got {text!r}") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"a rank of at least 1 is required, got {n}")
    return n


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
    p.add_argument("--rank", type=_rank, default=8,
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
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Recompute each layer's activations in the backward "
                        "pass instead of keeping them, trading time for memory. "
                        "Refused on Kimi K3 and DeepSeek-V4.1, whose layers "
                        "share state.")
    a = p.parse_args(argv)

    if a.grad_checkpoint and a.dropout > 0:
        # the compiled train step cannot replay a layer's dropout mask in
        # the backward recompute, so the recompute would see a fresh one
        print("error: --grad-checkpoint recomputes each layer under a fresh dropout mask; "
              "use it with --dropout 0", file=sys.stderr)
        return 2

    base, note, err = resolve_model_arg(a.model, a.config)
    if err is not None:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if note is not None:
        print(f"[config] {note}")

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
    err = probe_writable(adapter_out)
    if err is not None:
        print(f"error: --adapter-out is not writable: {err}", file=sys.stderr)
        return 2

    try:
        out, n = train_lora(
            os.path.abspath(os.path.expanduser(base)), a.data,
            adapter_out,
            iters=a.iters, batch_size=a.batch_size, num_layers=a.num_layers,
            rank=a.rank, scale=a.scale, dropout=a.dropout,
            learning_rate=a.learning_rate, max_seq_length=a.max_seq_length,
            val_batches=a.val_batches, steps_per_report=a.steps_per_report,
            steps_per_eval=a.steps_per_eval, seed=a.seed, hf_source=a.hf_source,
            grad_checkpoint=a.grad_checkpoint)
    except TrainRefused as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"[gmlx] wrote {n}-module LoRA adapter -> {out}")
    return 0
