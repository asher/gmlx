"""Hidden-state targets, an option off by default: the teacher pass
projects its final hidden state at every position through a seeded
random matrix to a few hundred dimensions and stores the sketch as the
``hidden`` field, and the trainer fits a linear map from the student's
final hidden state to the sketch at every shared boundary, with a
cosine or a squared-error loss beside the logit terms.

The projection is a Gaussian sketch scaled by 1 / sqrt(dim), so inner
products and distances between teacher states survive it up to a
sampling error of about 1 / sqrt(dim). The seed, the width and the layer
are in the manifest's ``hidden`` block, so a second cache of the same
teacher reproduces the same sketch. The learned map is a training-time
head: it lives in the checkpoint directory and never in the adapter."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

HIDDEN_FIELD = "hidden"
HS_MODES = ("cosine", "mse")


def projection_matrix(d_model: int, dim: int, seed: int) -> np.ndarray:
    """[d_model, dim] float32 Gaussian sketch, columns scaled by 1 / sqrt(dim)."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((d_model, dim)) / np.sqrt(dim)).astype(np.float32)


def hidden_block(d_model: int, dim: int, seed: int) -> dict:
    """The manifest block that names the sketch."""
    return {"layer": "final", "d_model": int(d_model), "dim": int(dim), "seed": int(seed),
            "dtype": "float16", "projection": "gaussian / sqrt(dim)"}


def sketch(hidden_btd, R) -> np.ndarray:
    """[B, T, dim] float16 sketch of the trunk output, evaluated."""
    import mlx.core as mx
    out = (hidden_btd.astype(mx.float32) @ R).astype(mx.float16)
    mx.eval(out)
    return np.asarray(out)


class HsHead:
    """The learned map from the student's hidden width to the sketch width
    and its optimizer state, with save and load beside a checkpoint."""

    def __init__(self, d_student: int, dim: int, seed: int, lr, weight_decay: float = 0.0):
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        # the map is built at the first --hs step, on a fresh run or a
        # resume, so its weights come from their own key and the run's
        # RNG stream is neither reseeded nor advanced
        self.module = nn.Linear.__new__(nn.Linear)
        nn.Module.__init__(self.module)
        bound = math.sqrt(1.0 / d_student)
        self.module.weight = mx.random.uniform(low=-bound, high=bound, shape=(dim, d_student),
                                               key=mx.random.key(seed + 7919))
        self.opt = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)
        self.dim = dim

    def update(self, grads: dict) -> None:
        import mlx.core as mx
        self.opt.update(self.module, grads)
        mx.eval(self.module.parameters(), self.opt.state)

    def save(self, d) -> None:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        mx.save_safetensors(str(d / "hs_head.safetensors"), dict(tree_flatten(self.module.parameters())))
        mx.save_safetensors(str(d / "hs_optimizer.safetensors"), dict(tree_flatten(self.opt.state)))

    def load(self, d) -> bool:
        import mlx.core as mx
        from mlx.utils import tree_unflatten
        p = d / "hs_head.safetensors"
        if not p.exists():
            return False
        params = mx.load(str(p))
        assert isinstance(params, dict)
        self.module.update(tree_unflatten(list(params.items())))
        ostate = mx.load(str(d / "hs_optimizer.safetensors"))
        assert isinstance(ostate, dict)
        self.opt.state = tree_unflatten(list(ostate.items()))
        return True


def hs_loss(pred, target, mode: str):
    """Mean over rows of 1 - cosine(pred, target), or the mean squared
    error over unit-normalized rows. Both are scale-free in the target,
    since the sketch's scale carries the teacher's residual-stream norm
    and not information the student's width could match."""
    import mlx.core as mx
    eps = mx.array(1e-6, dtype=mx.float32)
    p = pred.astype(mx.float32)
    t = target.astype(mx.float32)
    pn = p / mx.maximum(mx.linalg.norm(p, axis=-1, keepdims=True), eps)
    tn = t / mx.maximum(mx.linalg.norm(t, axis=-1, keepdims=True), eps)
    if mode == "cosine":
        return mx.mean(1.0 - mx.sum(pn * tn, axis=-1))
    return mx.mean(mx.sum((pn - tn) ** 2, axis=-1))


def hs_pass(hg_bnd, target, head: HsHead, mode: str) -> tuple[Any, Any, dict]:
    """(loss, d loss / d hg_bnd, d loss / d head params) on the gathered
    boundary hidden states, evaluated, outside any outer transform."""
    import mlx.core as mx

    def f(h, params):
        head.module.update(params)
        return hs_loss(head.module(h), target, mode)

    params = head.module.parameters()
    (loss, (dh, dparams)) = mx.value_and_grad(f, argnums=(0, 1))(hg_bnd.astype(mx.float32), params)
    mx.eval(loss, dh, dparams)
    return loss, dh, dparams
