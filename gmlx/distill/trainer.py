"""The trainer driver: the student load, LoRA or full-parameter setup, the
seeded batch iterator over one or more views, the head stage outside the
trunk's transform, checkpoints, and the adapter or checkpoint export.
``run_train`` is what ``gmlx distill train`` calls."""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gmlx.load.tokenizer import vocab_map_hash
from gmlx.tune.lora import LORA_KEYS, lora_scale, prepare_lora_student

from . import align as _align
from . import data as _data
from . import frames as _frames
from . import hidden as _hidden
from . import loss as _loss
from . import student as _student
from .constants import DEFAULT_KNOBS, GB, log
from .format import free_bytes, manifest_sha256, read_json, write_json_atomic
from .head import head_spec_from_model, log_bmask_from


@dataclass
class TrainOptions:
    views: list[str]
    student: str
    iters: int
    adapter_out: str | None = None
    out: str | None = None
    full: bool = False
    lora_rank: int = 16
    lora_scale: float | None = None
    lora_alpha: float | None = None
    lora_dropout: float = 0.0
    grad_checkpoint: bool = False
    lr: float = 1e-4
    batch_size: int = 8
    warmup: float = 0.05
    weight_decay: float | None = None
    clip: float = 1.0
    seed: int = 1
    loss: str = "bucketed"
    dk: float = DEFAULT_KNOBS["lambda_dk"]
    alm: float = DEFAULT_KNOBS["lambda_alm"]
    ce: float = DEFAULT_KNOBS["lambda_ce"]
    T_dk: float | None = None
    tau_alm: float | None = None
    gamma: float | None = None
    chunk: int = 512
    hs: float = 0.0
    hs_loss: str = "cosine"
    ckpt_dir: str | None = None
    resume: bool = False
    save_every: int = 200
    val_every: int = 200
    val_batches: int = 16
    report_every: int = 10
    report: str | None = None
    hf_source: str | None = None
    no_wired_limit: bool = False
    cache_limit_gb: float = 8.0
    extra: dict = field(default_factory=dict)


def is_gguf(path: str) -> bool:
    p = Path(path)
    return (p.is_file() and p.suffix == ".gguf") or (p.is_dir() and any(p.glob("*.gguf"))
                                                     and not (p / "tokenizer.json").exists())


def gguf_file(path: str) -> str:
    p = Path(path)
    return str(p if p.is_file() else sorted(p.glob("*.gguf"))[0])


def load_student(path: str, adapter: str | None, hf_source: str | None):
    """(model, config, tokenizer, kind) with kind "gguf" or "mlx"."""
    if is_gguf(path):
        model, cfg, tok = _student.load_gguf_student(gguf_file(path), adapter=adapter, hf_source=hf_source)
        return model, cfg, tok, "gguf"
    model, cfg, tok = _student.load_mlx_student(path, adapter_path=adapter)
    return model, cfg, tok, "mlx"


def make_schedule(lr: float, iters: int, warmup_frac: float):
    import mlx.optimizers as optim
    warm = max(1, int(iters * warmup_frac))
    warmup = optim.linear_schedule(0.0, lr, warm)
    cosine = optim.cosine_decay(lr, max(1, iters - warm))
    return optim.join_schedules([warmup, cosine], [warm])


def trainable_count(model) -> int:
    from mlx.utils import tree_flatten
    flat: Any = tree_flatten(model.trainable_parameters())
    return sum(int(np.prod(a.shape)) for _, a in flat)


def save_checkpoint(ckpt_dir: Path, tag: str, model, opt, state: dict, *, student_kind: str,
                    full: bool, base_cfg: dict | None, tokenizer=None) -> None:
    """Trainable parameters, optimizer state and the run state under
    ckpt_dir/tag, replaced atomically. A full fine-tune of an MLX student
    also writes a loadable checkpoint (weights, config, tokenizer)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    d = ckpt_dir / tag
    tmp = ckpt_dir / (tag + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    params: dict[str, Any] = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(tmp / "trainable.safetensors"), params)
    mx.save_safetensors(str(tmp / "optimizer.safetensors"), dict(tree_flatten(opt.state)))
    write_json_atomic(tmp / "state.json", state)
    if full and student_kind == "mlx" and base_cfg is not None:
        flat: Any = tree_flatten(model.parameters())
        allp: dict[str, Any] = {k: (v.astype(mx.bfloat16) if v.dtype == mx.float32 else v) for k, v in flat}
        mx.save_safetensors(str(tmp / "model.safetensors"), allp)
        write_json_atomic(tmp / "config.json", base_cfg)
        if tokenizer is not None:
            from gmlx.load.tokenizer import hf_inner
            hf_inner(tokenizer).save_pretrained(str(tmp))
    if d.exists():
        shutil.rmtree(d)
    os.replace(tmp, d)


def load_checkpoint(ckpt_dir: Path, tag: str, model, opt) -> dict:
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    d = ckpt_dir / tag
    params = mx.load(str(d / "trainable.safetensors"))
    assert isinstance(params, dict)
    model.update(tree_unflatten(list(params.items())))
    ostate = mx.load(str(d / "optimizer.safetensors"))
    assert isinstance(ostate, dict)
    opt.state = tree_unflatten(list(ostate.items()))
    return read_json(d / "state.json")


def run_train(opts: TrainOptions) -> int:
    """Returns 0 after the export, 2 on a refusal before training."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map

    if opts.lora_scale is not None and opts.lora_alpha is not None:
        log("[train] refuse: --lora-scale and --lora-alpha are two conventions for one multiplier, give one")
        return 2
    if not is_gguf(opts.student):
        log(f"[train] refuse: --student must be a GGUF file or a directory of GGUF shards: {opts.student}")
        return 2
    scale = lora_scale(opts.lora_rank,
                       scale=(2.0 if opts.lora_scale is None else opts.lora_scale) if opts.lora_alpha is None else None,
                       alpha=opts.lora_alpha)
    if not opts.views:
        log("[train] refuse: at least one --view is required")
        return 2
    view_dirs = [Path(v) for v in opts.views]
    views = [read_json(d / "view.json") for d in view_dirs]
    for d, v in zip(view_dirs, views):
        if manifest_sha256(Path(v["cache_dir"])) != v["cache_manifest_sha256"]:
            log(f"[train] refuse: the view's cache manifest hash does not match the cache on disk ({d})")
            return 2
    view_dir, view = view_dirs[0], views[0]
    tables = _align.load_tables(view_dir)
    for d, v in zip(view_dirs[1:], views[1:]):
        t2 = _align.load_tables(d)
        if (t2.teacher_hash, t2.student_hash, t2.V_T, t2.V_S, bool(v["identity"])) != \
                (tables.teacher_hash, tables.student_hash, tables.V_T, tables.V_S, bool(view["identity"])):
            log(f"[train] refuse: {d} is over another tokenizer pair than {view_dir}")
            return 2
        if (v.get("student_render_kwargs") or {}) != (view.get("student_render_kwargs") or {}):
            log(f"[train] refuse: {d} renders the student with other template kwargs than {view_dir}")
            return 2
    knobs = dict(view["knobs"], lambda_dk=opts.dk, lambda_alm=opts.alm, lambda_ce=opts.ce, loss_mode=opts.loss)
    if opts.T_dk is not None:
        knobs["T_dk"] = opts.T_dk
    if opts.tau_alm is not None:
        knobs["tau_alm"] = opts.tau_alm
    if opts.gamma is not None:
        knobs["gamma"] = opts.gamma
    if view["identity"]:
        knobs["lambda_alm"] = 0.0

    model, cfg, tokenizer, kind = load_student(opts.student, None, opts.hf_source)
    if opts.full and kind == "mlx":
        # f32 master weights: an AdamW step of lr on a bf16 weight rounds to
        # zero once |w| > lr * 2^8
        model.set_dtype(mx.float32)
        log("[train] full fine-tune: parameters cast to float32 (bf16 on save)")
    _frames.set_render_kwargs(tokenizer, view.get("student_render_kwargs") or {})
    if vocab_map_hash(tokenizer) != view["student_hash"]:
        log("[train] refuse: student tokenizer hash does not match the view")
        return 2
    inner = getattr(model, "language_model", model)
    head = head_spec_from_model(inner)
    if head.V != view["V_S"]:
        log(f"[train] refuse: student head width {head.V} != view V_S {view['V_S']}")
        return 2
    if not opts.no_wired_limit:
        try:
            mx.set_wired_limit(int(mx.device_info()["max_recommended_working_set_size"]))
        except (KeyError, RuntimeError, ValueError) as e:
            log(f"[train] warn: wired limit not set: {e}")
    # the batch shapes vary every step, so freed buffers of many sizes would
    # otherwise accumulate in MLX's cache up to the memory limit
    mx.set_cache_limit(int(opts.cache_limit_gb * GB))
    mx.random.seed(opts.seed)
    if opts.full:
        if kind != "mlx":
            log("[train] refuse: --full needs an MLX checkpoint student")
            return 2
        model.unfreeze()
        n_adapted = 0
    else:
        n_adapted = prepare_lora_student(model, rank=opts.lora_rank, scale=scale, dropout=opts.lora_dropout,
                                         keys=LORA_KEYS)
    from gmlx.tune.attention import install_training_attention
    from gmlx.tune.checkpoint import checkpoint_layers
    from gmlx.tune.gdn import install_training_gdn
    if opts.grad_checkpoint:
        log(f"[train] per-layer checkpointing on {checkpoint_layers(model)} layer classes")
    restore_attn = install_training_attention(model)
    log(f"[train] blocked attention: {getattr(restore_attn, 'count', 0)} attention modules patched")
    gdn_install = install_training_gdn(model)
    if gdn_install.count:
        log(f"[train] checkpointed gated delta training scan on {gdn_install.count} mlx-lm layers")
    model.train()
    head = head_spec_from_model(inner)
    wd = opts.weight_decay if opts.weight_decay is not None else (0.01 if opts.full else 0.0)
    opt = optim.AdamW(learning_rate=make_schedule(opts.lr, opts.iters, opts.warmup), weight_decay=wd)
    log(f"[train] {kind} student, {'full fine-tune' if opts.full else f'LoRA {n_adapted} modules'}, "
        f"{trainable_count(model) / 1e6:.2f}M trainable, same_tokenizer={view['identity']}, knobs={knobs}")

    G, Kp = tables.G, max(int(v["Kp"]) for v in views)
    readers = [_data.CacheReader(Path(v["cache_dir"])) for v in views]
    hidden_dim = None
    if opts.hs:
        if opts.hs_loss not in _hidden.HS_MODES:
            log(f"[train] refuse: --hs-loss must be one of {', '.join(_hidden.HS_MODES)}")
            return 2
        blocks = [(rd.manifest.get("gmlx_distill") or {}).get("hidden") for rd in readers]
        for d, blk in zip(view_dirs, blocks):
            if not blk:
                log(f"[train] refuse: --hs needs a cache with a hidden block (cache --hidden), {d} has none")
                return 2
        dims = {int(b["dim"]) for b in blocks}
        if len(dims) != 1:
            log(f"[train] refuse: the views' hidden sketches differ in width {sorted(dims)}")
            return 2
        hidden_dim = dims.pop()
        log(f"[train] hidden-state term: weight {opts.hs}, {opts.hs_loss} loss on a {hidden_dim}-dim sketch")
    loaders = [_data.ViewLoader(rd, tokenizer, tables, knobs=knobs, Kp=Kp, identity=bool(v["identity"]),
                                view_dir=d if any(d.glob("view-*.safetensors")) else None)
               for rd, v, d in zip(readers, views, view_dirs)]
    train_rows = [(vi, e["row"]) for vi, v in enumerate(views) for e in v["index"] if e["split"] == "train"]
    val_rows = [(vi, e["row"]) for vi, v in enumerate(views) for e in v["index"] if e["split"] == "val"]
    lengths = {(vi, e["row"]): e["n_student_tokens"] for vi, v in enumerate(views) for e in v["index"]}
    if len(views) > 1:
        counts = [sum(1 for vi, _ in train_rows if vi == i) for i in range(len(views))]
        log(f"[train] {len(views)} views mixed: train rows {counts} from {[str(d) for d in view_dirs]}")
    if len(train_rows) < opts.batch_size:
        log(f"[train] refuse: {len(train_rows)} train rows, fewer than --batch-size {opts.batch_size}")
        return 2
    it = _data.BatchIterator([lengths[r] for r in train_rows], opts.batch_size, opts.seed)

    def batch_rows(pairs):
        rvs = [loaders[vi].compile(r) for vi, r in pairs]
        rvs = [v for v in rvs if v is not None]
        return _data.collate(rvs, Kp, G) if rvs else None

    group_of = None if view["identity"] else mx.array(tables.group_of)
    hs_state: dict = {"head": None}

    def hs_head_for(d_student: int) -> _hidden.HsHead:
        """The learned map, created at the student's width on first use
        and restored from the last checkpoint on a resume."""
        if hs_state["head"] is None:
            assert hidden_dim is not None
            hh = _hidden.HsHead(d_student, hidden_dim, opts.seed,
                                make_schedule(opts.lr, opts.iters, opts.warmup), weight_decay=0.0)
            if opts.resume and hh.load(ckpt_dir / "last"):
                log("[train] hidden-state map restored from the last checkpoint")
            hs_state["head"] = hh
        return hs_state["head"]
    log_bmask = log_bmask_from(tables.bmask_S)

    def head_stage(batch):
        """Trunk forward and the head pass, outside any transform. Returns
        (loss, aux, positions, dh, dparams); see loss.head_pass for why the
        head never runs inside the trunk's transform."""
        hidden = _student.trunk_hidden(inner, batch["student_ids"])
        mx.eval(hidden)
        B, T, _d = hidden.shape
        hg = _loss.gather_positions(hidden, batch["positions"])
        loss, aux, dh, dparams = _loss.head_pass(hg, batch, head, group_of=group_of, G=G, Kp=Kp,
                                                 log_bmask=log_bmask, knobs=knobs, B=B, Tm1=T - 1, C=opts.chunk,
                                                 head_trainable=bool(opts.full))
        n_bnd = int(batch["n_bnd"])
        if opts.hs and n_bnd > 0 and "hidden_target" in batch:
            hh = hs_head_for(int(hg.shape[-1]))
            hs_val, dh_b, dW = _hidden.hs_pass(hg[:n_bnd], batch["hidden_target"], hh, opts.hs_loss)
            dh = mx.concatenate([dh[:n_bnd] + opts.hs * dh_b.astype(dh.dtype), dh[n_bnd:]], axis=0)
            loss = loss + opts.hs * hs_val
            mx.eval(loss, dh)
            aux["hs"] = hs_val
            aux["_hs_grads"] = tree_map(lambda g: opts.hs * g, dW)
        else:
            aux["hs"] = mx.zeros((), dtype=mx.float32)
        return loss, aux, batch["positions"], dh, dparams

    def trunk_loss(mdl, batch):
        hidden = _student.trunk_hidden(inner, batch["student_ids"])
        return _loss.trunk_surrogate(hidden, batch["_positions"], head, batch["_loss"], batch["_dh"],
                                     batch["_dparams"])

    vg = nn.value_and_grad(model, trunk_loss)

    def validate() -> float:
        model.eval()
        tot, ntok = 0.0, 0
        for i in range(0, min(len(val_rows), opts.val_batches * opts.batch_size), opts.batch_size):
            b = batch_rows(val_rows[i:i + opts.batch_size])
            if b is None:
                continue
            bm = _data.batch_to_mx({k: v for k, v in b.items() if not k.startswith("_")})
            loss, aux, _pos, _dh, _dp = head_stage(bm)
            n = int(aux["ntoks"])
            tot += float(loss) * n
            ntok += n
        model.train()
        return tot / max(ntok, 1)

    ckpt_dir = Path(opts.ckpt_dir) if opts.ckpt_dir else (Path(opts.out or ".") / "ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {"iteration": 0, "tokens": 0, "seed": opts.seed, "lr": opts.lr, "best_val": None,
             "knobs": knobs, "options": {k: v for k, v in vars(opts).items() if k != "extra"}}
    if opts.resume and (ckpt_dir / "last").exists():
        state = load_checkpoint(ckpt_dir, "last", model, opt)
        log(f"[train] resumed at iteration {state['iteration']}")
    est_ckpt = 2 * trainable_count(model) * 4 * 3
    if free_bytes(ckpt_dir) < 2 * est_ckpt:
        log(f"[train] refuse: free space under two checkpoints ({free_bytes(ckpt_dir) / GB:.2f} GB)")
        return 2
    log_rows = []
    t0 = time.perf_counter()
    skipped = 0
    step_walls: list[float] = []
    load_walls: list[float] = []
    base_cfg = cfg if isinstance(cfg, dict) else None
    for it_idx, rows in it.iterate(skip=state["iteration"]):
        if it_idx >= opts.iters:
            break
        tl0 = time.perf_counter()
        b = batch_rows([train_rows[i] for i in rows])
        load_walls.append(time.perf_counter() - tl0)
        if b is None or int(b["positions"].shape[0]) == 0:
            skipped += 1
            state["iteration"] = it_idx + 1
            continue
        bm = _data.batch_to_mx({k: v for k, v in b.items() if not k.startswith("_")})
        ts0 = time.perf_counter()
        loss, aux, bm["_positions"], bm["_dh"], bm["_dparams"] = head_stage(bm)
        bm["_loss"] = loss
        _value, grads = vg(model, bm)
        if opts.clip:
            grads, _norm = optim.clip_grad_norm(grads, opts.clip)
        opt.update(model, grads)
        mx.eval(model.trainable_parameters(), opt.state, loss)
        if "_hs_grads" in aux:
            hs_state["head"].update(aux.pop("_hs_grads"))
        step_walls.append(time.perf_counter() - ts0)
        state["iteration"] = it_idx + 1
        state["tokens"] += int(aux["ntoks"])
        if (it_idx + 1) % opts.report_every == 0:
            rec = {"it": it_idx + 1, "loss": float(loss), "dk": float(aux["dk"]), "alm": float(aux["alm"]),
                   "ce": float(aux["ce"]), "hs": float(aux["hs"]), "floored": int(aux["floored"]),
                   "tokens": state["tokens"],
                   "lr": float(opt.learning_rate), "wall_s": time.perf_counter() - t0,
                   "peak_gb": mx.get_peak_memory() / GB,
                   "active_gb": mx.get_active_memory() / GB, "cache_gb": mx.get_cache_memory() / GB,
                   "step_ms": 1e3 * float(np.median(step_walls[-opts.report_every:])),
                   "load_ms": 1e3 * float(np.median(load_walls[-opts.report_every:]))}
            log_rows.append(rec)
            log(f"[train] it {rec['it']} loss {rec['loss']:.4f} dk {rec['dk']:.4f} alm {rec['alm']:.4f} "
                f"ce {rec['ce']:.4f}" + (f" hs {rec['hs']:.4f}" if opts.hs else "") + " "
                f"floored {rec['floored']} lr {rec['lr']:.2e} "
                f"{rec['tokens'] / max(rec['wall_s'], 1e-9):.0f} tok/s step {rec['step_ms']:.0f} ms "
                f"load {rec['load_ms']:.0f} ms peak {rec['peak_gb']:.1f} GB "
                f"active {rec['active_gb']:.1f} cache {rec['cache_gb']:.1f}")
        if (it_idx + 1) % opts.val_every == 0 or it_idx + 1 == opts.iters:
            v = validate()
            log_rows.append({"it": it_idx + 1, "val": v})
            best = state['best_val']
            log(f"[train] it {it_idx + 1} val {v:.4f} (best {best:.4f})" if best is not None
                else f"[train] it {it_idx + 1} val {v:.4f}")
            if state["best_val"] is None or v < state["best_val"]:
                state["best_val"] = v
                save_checkpoint(ckpt_dir, "best", model, opt, state, student_kind=kind, full=opts.full,
                                base_cfg=base_cfg, tokenizer=tokenizer)
                if hs_state["head"] is not None:
                    hs_state["head"].save(ckpt_dir / "best")
        if (it_idx + 1) % opts.save_every == 0 or it_idx + 1 == opts.iters:
            save_checkpoint(ckpt_dir, "last", model, opt, state, student_kind=kind, full=opts.full,
                            base_cfg=base_cfg, tokenizer=tokenizer)
            if hs_state["head"] is not None:
                hs_state["head"].save(ckpt_dir / "last")
    restore_attn()
    log(f"[train] done: {state['iteration']} steps, {state['tokens']} tokens, {skipped} skipped, "
        f"{time.perf_counter() - t0:.0f}s")
    if not opts.full and kind == "gguf" and opts.adapter_out:
        from gmlx.load.preflight import preflight
        from gmlx.tune.lora import save_trained_adapter
        n = save_trained_adapter(inner, cfg, base_arch=preflight(gguf_file(opts.student)).arch,
                                 out_path=opts.adapter_out, rank=opts.lora_rank, scale=scale, keys=LORA_KEYS)
        log(f"[train] wrote {opts.adapter_out} ({n} modules)")
    elif not opts.full and kind == "mlx" and opts.out:
        out = Path(opts.out)
        out.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
        write_json_atomic(out / "adapter_config.json", {
            "fine_tune_type": "lora", "num_layers": len(getattr(inner, "layers", [])),
            "lora_parameters": {"rank": opts.lora_rank, "scale": scale, "alpha": scale * opts.lora_rank,
                                "dropout": opts.lora_dropout, "keys": list(LORA_KEYS)}})
        log(f"[train] wrote {out}/adapters.safetensors")
    elif opts.full and opts.out:
        out = Path(opts.out)
        if (ckpt_dir / "last").exists():
            out.mkdir(parents=True, exist_ok=True)
            for f in (ckpt_dir / "last").iterdir():
                if f.is_file() and f.name not in ("state.json", "optimizer.safetensors", "trainable.safetensors"):
                    shutil.copy(f, out / f.name)
            gen_cfg = Path(opts.student) / "generation_config.json"
            if gen_cfg.is_file() and not (out / gen_cfg.name).exists():
                shutil.copy(gen_cfg, out / gen_cfg.name)
            log(f"[train] wrote full checkpoint to {out}")
    if opts.report:
        write_json_atomic(Path(opts.report), {
            "state": state, "log": log_rows,
            "view": str(view_dir) if len(view_dirs) == 1 else [str(d) for d in view_dirs],
            "timing": {"step_ms_median": 1e3 * float(np.median(step_walls)) if step_walls else None,
                       "load_ms_median": 1e3 * float(np.median(load_walls)) if load_walls else None,
                       "peak_gb": mx.get_peak_memory() / GB}})
    return 0
