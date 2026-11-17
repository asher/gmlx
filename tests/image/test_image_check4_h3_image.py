"""Check 4 of the image acceptance gate for MiniMax-H3-Image: the family's
sampling row against the PDD grid ComfyUI samples on, and the heads each
step reads against ComfyUI's PDD rule.

ComfyUI builds the grid from the GGUF's config key, as pdd_config.json
holds it, and its `simple` schedule at 6 steps lands up to 9e-4 off the
grid. gmlx samples the grid itself, so the schedule its pipeline hands
the sampler, and its final zero, equal ComfyUI's grid bit for bit. Each
step reads the heads ComfyUI picks from the sigma and the sigma the step
goes to: one grid interval with weight 1 on the grid and on the simple
schedule at 6 steps, and two intervals with their float32 dt weights on
each step of the simple schedule at 3. On those three steps the head
gives the real _pdd_head's output on a seeded tiny head bank. The arrays
are small and committed, so this runs on the CPU tier.
scripts/image_oracle/check4_h3_image.py writes them."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from image_fixtures import fixture_view, load_json
from oracle_fixture import committed_fixture

from gmlx.image import api
from gmlx.image.families.h3_image import family as hi
from gmlx.image.families.h3_image import model as him

NAME = "check4_h3_image"
# A bound on how far ComfyUI's simple schedule at 6 steps lands off the
# grid, a fact of the fixture. No step takes a nearby sigma as the grid's:
# each reads the heads it spans by ComfyUI's rule.
GRID_TOL = 2e-3
SCHEDULES = ("simple_6", "grid", "simple_3")


@pytest.fixture(scope="module")
def check4():
    return committed_fixture(NAME)


def test_the_fixture_reads_the_committed_pdd_config(check4):
    _, m = check4
    want = load_json("manifest.json")["h3_image_pdd_config.json"]
    got = m["files"]["pdd_config"]
    assert (got["size"], got["sha256"]) == (want["size"], want["sha256"])
    assert m["inputs"]["steps"] == hi.STEPS
    assert m["inputs"]["off_grid_steps"] == 3
    assert m["inputs"]["pdd_sigmas"] == list(hi.PDD_SIGMAS)


def test_the_pipelines_schedule_is_comfyuis_grid_bit_for_bit(check4):
    arrays, m = check4
    assert m["tolerances"]["grid"] == "bitwise"
    pipe = SimpleNamespace(family=hi.FAMILY, variant="pruned")
    got = api.ImagePipeline.sigmas(pipe, hi.STEPS)
    assert got.dtype == np.float32
    assert got.tobytes() == arrays["grid"].tobytes()
    assert hi.FAMILY.sampling_row("pruned").steps == hi.STEPS


def test_the_simple_schedule_lands_near_the_grid(check4):
    arrays, m = check4
    off = float(np.abs(arrays["simple_6"] - arrays["grid"]).max())
    assert off == pytest.approx(m["results"]["simple_off_grid"])
    assert 0 < off < GRID_TOL


def _misses(arrays, m, sched) -> list:
    """The steps of ``sched`` whose heads, start, stop and weights, or
    whose timestep differ from ComfyUI's."""
    s = arrays[sched]
    out = []
    for i, ((start, stop), w, ts) in enumerate(zip(
            arrays[f"heads_{sched}"].tolist(),
            m["results"]["head_weights"][sched],
            arrays[f"timesteps_{sched}"])):
        got = hi.step_heads(s[i], s[i + 1])
        # ComfyUI takes head 0 whole, so its weights are those past it.
        got = (*got[:2], got[2][max(start, 1) - start:])
        if got != (start, stop, tuple(w)):
            out.append((i, got, (start, stop, w)))
        if not (np.float32(s[i]) * np.float32(1000) == ts
                and hi.model_sigma(s[i]) == float(ts) / 1000.0):
            out.append((i, "timestep", float(ts)))
    return out


@pytest.mark.parametrize("sched", SCHEDULES)
def test_each_step_reads_comfyuis_heads(check4, sched):
    """The heads of each step, and the timestep the DiT reads, are
    ComfyUI's on its simple schedule, on the grid, and on the simple
    schedule at 3 steps, whose steps span two grid intervals."""
    arrays, m = check4
    assert m["tolerances"]["heads"] == "exact"
    assert not _misses(arrays, m, sched)
    spans = [b - a for a, b in arrays[f"heads_{sched}"].tolist()]
    assert spans == ([2] * 3 if sched == "simple_3" else [1] * hi.STEPS)


def test_the_last_grid_point_at_or_above_sigma_misses(check4, monkeypatch):
    """A plausible wrong rule, through the same comparison: it reads the
    grid point before the nearest on the simple steps just above it."""
    arrays, m = check4
    monkeypatch.setattr(hi, "grid_point", lambda s: max(
        (k for k, g in enumerate(hi.PDD_SIGMAS) if g >= s), default=0))
    missed = _misses(arrays, m, "simple_6")
    print(f"h3 image check4 wrong rule: {len(missed)} of {hi.STEPS} "
          f"simple steps missed")
    assert missed


class _Through(nn.Module):
    def __call__(self, x):
        return x


def _bank_model(arrays) -> him.H3ImageDiT:
    """A tiny DiT whose final layer is the record's head bank, its norm
    passed through and its adaLN rows zero, so the head reads the
    record's input as ComfyUI's _pdd_head did."""
    w = arrays["head_bank_weight"]
    cfg = {**hi.synth_config(fixture_view(
        "h3_image_pruned_ud_q4_k_xl.header.json")),
        "num_layers": 0, "token_refiner_num_layers": 0,
        "hidden_size": w.shape[1], "num_attention_heads": 1,
        "attention_head_dim": w.shape[1], "ffn_hidden_size": 32,
        "latents_dim": w.shape[0] // hi.STEPS // 4}
    model = him.H3ImageDiT(cfg)
    fl = model.final_layer
    assert fl.video_out.weight.shape == w.shape
    fl.norm = _Through()
    lin = fl.adaln_proj.linear
    lin.weight, lin.bias = mx.zeros_like(lin.weight), mx.zeros_like(lin.bias)
    fl.video_out.weight = mx.array(w)
    fl.video_out.bias = mx.array(arrays["head_bank_bias"])
    return model


def _rel(got: mx.array, want: np.ndarray) -> float:
    got = np.array(got, np.float64)
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def test_the_head_on_steps_that_span_intervals_is_comfyuis(check4):
    """Each step of the simple schedule at 3 spans two grid intervals. On
    the seeded bank the head gives the real _pdd_head's output within
    float32 rounding, and with each spanned head read at weight 1 it
    misses."""
    arrays, m = check4
    tol = m["tolerances"]["head_out_rel"]
    model = _bank_model(arrays)
    s = arrays["simple_3"]
    h, want = arrays["head_in_simple_3"], arrays["head_out_simple_3"]
    plan = model.plan(1, 2, 2 * h.shape[1])
    assert plan.target.stop - plan.target.start == h.shape[1]
    pad = mx.zeros((1, plan.target.start, h.shape[2]))
    rel, whole = [], []
    # On the CPU: the GPU's float32 matmul can run at TF32, 7e-4 off here.
    with mx.stream(mx.cpu):
        for i in range(len(s) - 1):
            mod = model.modulation(s[i], 1, s[i + 1])
            x = mx.concatenate([pad, mx.array(h[i])[None]], axis=1)
            rel.append(_rel(model.head(x, mod, plan), want[i]))
            ones = dataclasses.replace(mod, heads=tuple(
                (k, 1.0) for k, _ in mod.heads))
            whole.append(_rel(model.head(x, ones, plan), want[i]))
    print(f"h3 image check4 head rel L2 {rel}; spanned heads at weight 1 "
          f"{whole}")
    assert max(rel) <= tol
    assert min(whole) > 100 * tol
