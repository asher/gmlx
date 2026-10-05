"""The gate between the in-op LoRA epilogue and the stock-op delta on a
dense adapter layer: GMLX_LORA_EPILOGUE, the NAX flag, the adapter rank
and the activation rows. The routing rows are CPU-safe, since the base
call is a fake. The agreement row runs both paths on a K-quant weight."""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import gmlx.load.modules as modules

D = 64


@pytest.fixture
def routed(monkeypatch):
    """Set the mode and the NAX flag, then report whether a call of the
    given rows and rank handed its tables to the base."""
    from mlx_kquant.nn import KQuantLinear

    calls = []

    def fake_call(self, x, lora=None):
        calls.append(lora is not None)
        return mx.zeros(x.shape[:-1] + (D,), x.dtype)

    monkeypatch.setattr(KQuantLinear, "__call__", fake_call)
    monkeypatch.setattr(modules, "_KQ_LORA_EPILOGUE", True)

    def route(mode: str, nax: bool, rows: int, rank: int) -> bool:
        monkeypatch.setattr(modules, "_LORA_EPILOGUE_MODE", mode)
        monkeypatch.setattr(modules, "_NAX", nax)
        wrap = modules.LoRAKQuantLinear(
            KQuantLinear(D, D, False, "q8_0"), mx.ones((rank, D)),
            mx.ones((D, rank)), scale=1.0)
        wrap(mx.zeros((rows, D), mx.bfloat16))
        return calls.pop()

    return route


def test_auto_on_nax_follows_the_sweep_table(routed):
    table = modules._LORA_EPILOGUE_GATE[True]
    assert table == tuple(sorted(table)) and len(table) >= 1
    for max_rank, max_rows in table:
        assert routed("auto", True, max_rows, max_rank) is True
        assert routed("auto", True, max_rows + 1, max_rank) is False
    # A decode row at a small rank rides the kernel, and an image-sized or
    # prefill-sized call never does.
    assert routed("auto", True, 1, 8) is True
    assert routed("auto", True, 4608, 8) is False
    assert routed("auto", True, 1, table[-1][0] + 1) is False
    # The rows are every leading dimension of the call.
    wins = modules.lora_epilogue_wins
    assert wins(4 * 4, 16) is True and wins(4 * 5, 16) is False


# Swept cells the NAX rule sends to the slower path by more than the A
# versus A band, as (in, out, rows, rank) -> the time lost on that layer.
# One row bound per rank serves a wide and a narrow output: the first three
# are the wide output's price, and the rest the narrow outputs'.
ACCEPTED = {
    (3840, 11520, 8, 32): 0.212,
    (3840, 11520, 32, 8): 0.104,
    (3840, 11520, 32, 4): 0.030,
    (4096, 4096, 32, 16): 0.070,
    (4096, 4096, 64, 4): 0.043,
    (10240, 3840, 32, 16): 0.044,
    (10240, 3840, 64, 4): 0.036,
}


def _sweep() -> dict:
    path = (Path(__file__).resolve().parents[1] / "fixtures" / "lora"
            / "epilogue_gate_sweep_nax.json")
    return json.loads(path.read_text())


def test_the_nax_rule_routes_every_swept_cell_to_its_faster_path(
        monkeypatch):
    """Every cell where one path wins by more than the band goes to that
    path, or is an accepted exception at its measured cost."""
    monkeypatch.setattr(modules, "_LORA_EPILOGUE_MODE", "auto")
    monkeypatch.setattr(modules, "_NAX", True)
    sweep = _sweep()
    assert sweep["head"]["nax"] is True and len(sweep["cells"]) == 252
    band, lost = sweep["band"], {}
    for inp, out, rows, rank, ratio in sweep["cells"]:
        epilogue = modules.lora_epilogue_wins(rows, rank)
        cost = ratio - 1 if epilogue else 1 / ratio - 1
        if cost > band:
            lost[(inp, out, rows, rank)] = cost
    assert set(lost) == set(ACCEPTED), sorted(set(lost) ^ set(ACCEPTED))
    for cell, cost in lost.items():
        assert abs(cost - ACCEPTED[cell]) < 1e-3, (cell, cost)
    assert max(lost.values()) < 0.22


def test_no_rule_of_one_row_bound_per_rank_has_a_smaller_worst_loss():
    """The table is the minimax rule over the swept shapes: per rank, no
    other row bound loses less in its worst cell."""
    sweep = _sweep()
    rows_swept = sorted({c[2] for c in sweep["cells"]})
    table = dict(modules._LORA_EPILOGUE_GATE[True])

    def worst(rank: int, bound: int) -> float:
        return max((r - 1 if rows <= bound else 1 / r - 1)
                   for _, _, rows, k, r in sweep["cells"] if k == rank)

    def bound_of(rank: int) -> int:
        return next((b for k, b in sorted(table.items()) if rank <= k), 0)

    for rank in sorted({c[3] for c in sweep["cells"]}):
        best = min(worst(rank, b) for b in [0] + rows_swept)
        assert worst(rank, bound_of(rank)) <= best + 1e-9, rank


def test_a_gpu_class_with_no_sweep_keeps_the_epilogue(routed):
    assert False not in modules._LORA_EPILOGUE_GATE
    for rows, rank in ((1, 4), (64, 16), (4608, 128), (16896, 256)):
        assert routed("auto", False, rows, rank) is True


def test_always_and_never_override_the_table(routed):
    for nax in (True, False):
        for rows, rank in ((1, 4), (4608, 128)):
            assert routed("always", nax, rows, rank) is True
            assert routed("never", nax, rows, rank) is False
    assert modules._LORA_EPILOGUE_MODES == ("auto", "always", "never")


def test_the_mode_never_turns_on_a_kernel_the_build_lacks(routed,
                                                           monkeypatch):
    monkeypatch.setattr(modules, "_KQ_LORA_EPILOGUE", False)
    monkeypatch.setattr(modules, "_LORA_EPILOGUE_MODE", "always")
    from mlx_kquant.nn import KQuantLinear

    seen = []
    monkeypatch.setattr(KQuantLinear, "__call__",
                        lambda self, x: seen.append(1) or mx.zeros(
                            x.shape[:-1] + (D,), x.dtype))
    wrap = modules.LoRAKQuantLinear(
        KQuantLinear(D, D, False, "q8_0"), mx.ones((4, D)),
        mx.ones((D, 4)), scale=1.0)
    assert wrap(mx.zeros((1, D), mx.bfloat16)).shape == (1, D) and seen


def test_the_env_value_falls_back_to_auto(monkeypatch):
    from gmlx import envflags

    assert modules._LORA_EPILOGUE_MODE in modules._LORA_EPILOGUE_MODES
    for raw, want in (("never", "never"), ("ALWAYS", "always"),
                      ("auto", "auto"), ("sometimes", "auto")):
        monkeypatch.setenv("GMLX_LORA_EPILOGUE", raw)
        assert envflags.env_choice("GMLX_LORA_EPILOGUE", "auto",
                                   modules._LORA_EPILOGUE_MODES) == want
    monkeypatch.delenv("GMLX_LORA_EPILOGUE")
    assert envflags.env_choice("GMLX_LORA_EPILOGUE", "auto",
                               modules._LORA_EPILOGUE_MODES) == "auto"


def test_both_paths_give_the_same_delta(monkeypatch):
    kq = pytest.importorskip("mlx_kquant")
    if not getattr(kq, "HAS_LORA_EPILOGUE", False):
        pytest.skip("this mlx-kquant build has no LoRA epilogue")
    from mlx_kquant.nn import KQuantLinear

    rng = np.random.default_rng(0)
    inp, out, rank = 256, 96, 8
    base = KQuantLinear(inp, out, False, "q4_k")
    base.weight, base.scales = kq.quantize(mx.array(
        (rng.standard_normal((out, inp)) / 16).astype(np.float32)), "q4_k")
    wrap = modules.LoRAKQuantLinear(
        base, mx.array((rng.standard_normal((rank, inp)) / 16)
                       .astype(np.float32)),
        mx.array(rng.standard_normal((out, rank)).astype(np.float32)),
        scale=0.5)
    x = mx.array(rng.standard_normal((3, inp)).astype(np.float32)).astype(
        mx.bfloat16)
    got = {}
    for mode in ("always", "never"):
        monkeypatch.setattr(modules, "_LORA_EPILOGUE_MODE", mode)
        got[mode] = np.array(wrap(x).astype(mx.float32), np.float64)
    plain = np.array(base(x).astype(mx.float32), np.float64)
    diff = np.linalg.norm(got["always"] - got["never"])
    assert np.linalg.norm(got["never"] - plain) > 0.1 * np.linalg.norm(plain)
    assert diff <= 2e-2 * np.linalg.norm(got["never"])


def test_rows_mode_takes_the_same_row_layouts_on_both_paths(monkeypatch):
    """Rows as (B, L, D), (B * L, D) and (1, B * L, D): the plain path
    scales each input row by its batch row's factor, and the epilogue
    path hands the kernel the same factor per row."""
    from mlx_kquant.nn import KQuantLinear

    from gmlx import lora_rows

    seen = []

    def fake_call(self, x, lora=None):
        seen.append(lora)
        return mx.zeros(x.shape[:-1] + (D,), x.dtype)

    monkeypatch.setattr(KQuantLinear, "__call__", fake_call)
    monkeypatch.setattr(modules, "_KQ_LORA_EPILOGUE", True)
    rng = np.random.default_rng(1)
    a = mx.array(rng.standard_normal((4, D)).astype(np.float32))
    b = mx.array(rng.standard_normal((D, 4)).astype(np.float32))
    wrap = modules.LoRAKQuantLinear(KQuantLinear(D, D, False, "q8_0"), a, b,
                                    scale=0.5, slot=0)
    wrap.add_slot(a * 0.5, b, 1.0, 1)
    batch, tokens = 2, 3
    x = rng.standard_normal((batch, tokens, D)).astype(np.float32)
    scales = np.array([[2.0, 1.0], [0.0, 3.0]], np.float32)
    # The base is zero, so the output is the delta: each slot's product
    # times its batch row's factor.
    one = x @ np.array(a).T @ np.array(b).T
    want = (one * 0.5 * scales[:, 0, None, None]
            + one * 0.5 * scales[:, 1, None, None]).reshape(-1, D)
    layouts = ((batch, tokens, D), (batch * tokens, D),
               (1, batch * tokens, D))
    lora_rows.configure("rows", 2)
    try:
        with mx.stream(mx.cpu):
            for mode in ("never", "always"):
                monkeypatch.setattr(modules, "_LORA_EPILOGUE_MODE", mode)
                for shape in layouts:
                    lora_rows.set_rows(scales)
                    seen.clear()
                    y = wrap(mx.array(x.reshape(shape)))
                    got = np.array(y).reshape(-1, D)
                    if mode == "never":
                        assert seen == [None]
                        np.testing.assert_allclose(got, want, rtol=2e-5,
                                                   atol=2e-5)
                        continue
                    # float32 activations keep the plain ops, so the
                    # epilogue rows are read from a bfloat16 call.
                    seen.clear()
                    wrap(mx.array(x.reshape(shape)).astype(mx.bfloat16))
                    rows = seen[0][2]
                    assert np.array(rows).tolist() == [2.0] * 3 + [0.0] * 3
    finally:
        lora_rows.set_rows(None)
        lora_rows.configure("static", 1)
