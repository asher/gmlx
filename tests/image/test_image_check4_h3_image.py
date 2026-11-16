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
each step of the simple schedule at 3. The arrays are small and
committed, so this runs on the CPU tier.
scripts/image_oracle/check4_h3_image.py writes them."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from image_fixtures import load_json
from oracle_fixture import committed_fixture

from gmlx.image import api
from gmlx.image.families.h3_image import family as hi

NAME = "check4_h3_image"
# How far off the grid a step route and the pack take a sigma as the
# grid's (design note section 2), over ComfyUI's simple schedule.
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
