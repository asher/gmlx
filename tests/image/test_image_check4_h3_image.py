"""Check 4 of the image acceptance gate for MiniMax-H3-Image: the family's
sampling row against the PDD grid ComfyUI samples on, and the head each
step reads against ComfyUI's PDD rule.

ComfyUI builds the grid from the GGUF's config key, as pdd_config.json
holds it, and its `simple` schedule at 6 steps lands up to 9e-4 off the
grid. gmlx samples the grid itself, so the schedule its pipeline hands
the sampler, and its final zero, equal ComfyUI's grid bit for bit; on
both schedules each step reads the head ComfyUI picks, one grid interval
with weight 1. The arrays are small
and committed, so this runs on the CPU tier.
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


@pytest.fixture(scope="module")
def check4():
    return committed_fixture(NAME)


def test_the_fixture_reads_the_committed_pdd_config(check4):
    _, m = check4
    want = load_json("manifest.json")["h3_image_pdd_config.json"]
    got = m["files"]["pdd_config"]
    assert (got["size"], got["sha256"]) == (want["size"], want["sha256"])
    assert m["inputs"]["steps"] == hi.STEPS
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
    """The steps of ``sched`` whose head, or whose timestep, differ from
    ComfyUI's: its pick of one grid interval with weight 1."""
    out = []
    for i, (s, (start, stop), w, ts) in enumerate(zip(
            arrays[sched][:-1], arrays[f"heads_{sched}"].tolist(),
            m["results"]["head_weights"][sched],
            arrays[f"timesteps_{sched}"])):
        if (hi.step_head(s), start + 1, [1.0]) != (start, stop, w):
            out.append((i, hi.step_head(s), (start, stop, w)))
        if not (np.float32(s) * np.float32(1000) == ts
                and hi.model_sigma(s) == float(ts) / 1000.0):
            out.append((i, "timestep", float(ts)))
    return out


@pytest.mark.parametrize("sched", ("simple_6", "grid"))
def test_each_step_reads_comfyuis_head(check4, sched):
    """The head of each step, and the timestep the DiT reads, are
    ComfyUI's on its simple schedule and on the grid."""
    arrays, m = check4
    assert m["tolerances"]["heads"] == "exact"
    assert not _misses(arrays, m, sched)
    sig = arrays[sched][:-1]
    assert [hi.step_head(s) for s in sig] == list(range(hi.STEPS))


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
