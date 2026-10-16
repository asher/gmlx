"""Check 6 of the image acceptance gate for FLUX.3 klein, on latents: a
whole sampling run on injected noise, for the distilled flow map and for
Base. gmlx samples from the seed's noise with the variant's defaults on
the oracle's conditioning, the empty prompt's for the unconditional row
of Base. The seed's noise must equal the oracle's, and the schedule and
the times the DiT saw must be the reference's.

At 512 the run is the model as it ships, in bf16, and the latents after
each step the oracle kept and after the last must stay within twice the
oracle's own bf16 against fp32 gap after that step. A flow map that goes
to its own time instead of the next, and a Base run whose unconditional
row is the prompt's, fail the kept steps.

At 1024 the gap of one bf16 run lies inside the spread of valid bf16
roundings, so the gate is the exact run: check 3's exact arithmetic, held
to EXACT_REL_L2 of the oracle's fp32 latents after each kept step and
after the last. The time embedding without its inner bf16 rounding, a
flow map to its own time, Base with the prompt as its unconditional row,
and the image positions at rope scale 1 each fail it. The bf16 run at
1024 must stay finite, and its readings print as a share of the 512 gate
of each step, as the manifest records them.

Skipped unless the manifests of scripts/image_oracle/check6_flux3_klein.py
are committed, GMLX_IMAGE_FIXTURES holds their arrays and
KQUANT_TEST_GGUF_DIR the DiT GGUFs the oracle read."""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import flux3_gates as fg
import mlx.core as mx
import numpy as np
import pytest
from block_gates import exact_process, rerun_exact
from image_fixtures import image_dits
from oracle_fixture import ORACLE, committed, fixture, fx

# A Mac of 16 GB, as most users run (tests/host_box.py).
pytestmark = pytest.mark.host_gb(16)

NAMES = {"distilled": "check6_flux3_klein", "base": "check6_flux3_klein_base"}
GATE_SIZE = 512
EXACT_SIZE = 1024
SIZES = (GATE_SIZE, EXACT_SIZE)
EXACT_REL_L2 = 1e-3
REFERENCE_COMMIT = "45e57c9fb06f9751cdced29a65c5d2171003be05"
CASES = [(v, s) for v in NAMES for s in SIZES]


def _name(variant: str, size: int) -> str:
    return f"{NAMES[variant]}_{size}"


def _have(variant: str, size: int) -> bool:
    return (ORACLE / f"{_name(variant, size)}.manifest.json").is_file()


def _manifest(variant: str, size: int) -> dict:
    if not _have(variant, size):
        pytest.skip(f"{_name(variant, size)}.manifest.json is not "
                    "committed: run scripts/image_oracle/check6_flux3_klein.py"
                    " on the FLUX.3 klein GGUF and commit the manifest")
    return committed(_name(variant, size))


def _bits(a) -> list:
    return np.asarray(a, np.float32).view(np.uint32).tolist()


def _kept(m: dict) -> list:
    return m["inputs"]["kept_steps"] + [m["inputs"]["steps"]]


def _gates(variant: str) -> dict:
    """The bf16 gate of the latents after each kept step and after the
    last, at the gate size."""
    m = _manifest(variant, GATE_SIZE)
    by_step = m["tolerances"]["by_step_rel_l2"]
    assert len(by_step) == m["inputs"]["steps"]
    return {k: by_step[k - 1] for k in _kept(m)}


class _Pipes:
    """The pipeline of a fixture's file, one loaded at a time."""

    def __init__(self, root):
        self.root, self.path, self.pipe = root, None, None

    def release(self) -> None:
        self.pipe = self.path = None
        mx.clear_cache()

    def __call__(self, m: dict):
        from gmlx.image import api

        want = m["files"]["dit"]
        if self.path is not None and Path(self.path).name == want["name"]:
            return self.pipe
        self.release()
        for p in image_dits(self.root, "flux3_klein"):
            if Path(p).name == want["name"] and \
                    os.path.getsize(p) == want["size"]:
                break
        else:
            pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
        if fx.file_identity(p)["sha256"] != want["sha256"]:
            pytest.fail(f"{p} is not the file the oracle read")
        self.pipe = api.load(p, log=lambda *_: None)
        self.path = p
        return self.pipe


@pytest.fixture(scope="module")
def pipes(gguf_dir):
    return _Pipes(gguf_dir)


def _conds(arrays: dict, variant: str, size: int):
    from gmlx.image.conditioning import Conditioning

    cond = Conditioning(mx.array(arrays["features"]),
                        f"check6 flux3 {variant} {size}")
    neg = None
    if variant == "base":
        neg = Conditioning(mx.array(arrays["features_empty"]),
                           f"check6 flux3 {variant} {size} empty")
    return cond, neg


def _run(pipe, arrays: dict, m: dict, variant: str, size: int,
         stop: int | None = None, neg=False) -> tuple[dict, bool]:
    """The relative L2 of the latents after each kept step and after the
    last against the oracle's fp32 latents, and whether the latents
    stayed finite after every step. ``stop`` ends the run after that
    many steps."""
    from gmlx.image import api

    steps = m["inputs"]["steps"]
    want = {k: arrays[f"latents_step{k}"] for k in m["inputs"]["kept_steps"]}
    want[steps] = arrays["latents"]
    cond, uncond = _conds(arrays, variant, size)
    if neg:
        uncond = cond
    got: dict = {}
    finite = [True]

    def on_step(i, x):
        a = np.array(x)
        finite[0] = finite[0] and bool(np.isfinite(a).all())
        if i + 1 in want:
            got[i + 1] = fx.rel_l2(a, want[i + 1])

    sigmas = None
    if stop is not None:
        sigmas = pipe.sigmas(steps)[:stop + 1]
    api.sample(pipe, cond, neg=uncond, height=size, width=size,
               noise=mx.array(arrays["noise"]), sigmas=sigmas,
               on_step=on_step)
    return got, finite[0]


def _line(label: str, got: dict, gates: dict) -> str:
    return f"{label}: " + ", ".join(
        f"step {k} {v:.2e} ({v / gates[k]:.2f} of its gate)"
        for k, v in sorted(got.items()))


def test_the_manifests_follow_the_rule():
    """The committed manifests alone: each run is the reference's on its
    commit at pipeline.MODELS' steps and guidance and its alpha, the
    pipeline encodes the empty prompt first for Base, every gate at the
    gate size is twice the gap beside it, and the exact size holds the
    exact tolerance and the bf16 readings of each kept step."""
    from gmlx.image.registry import get_family

    fam = get_family("flux3_klein")
    seen = 0
    for variant, size in CASES:
        if not _have(variant, size):
            continue
        seen += 1
        m = committed(_name(variant, size))
        i, r = m["inputs"], m["results"]["run"]
        row = fam.sampling[variant]
        assert i["reference_commit"] == REFERENCE_COMMIT
        assert (i["variant"], i["size"], i["refs"]) == (variant, size, 0)
        assert i["flow_map"] is (variant == "distilled")
        assert i["steps"] == row.steps
        assert (i["guidance"] or 1.0) == row.cfg
        assert i["alpha"] == fam.constants.values["alpha"]
        assert r["prompts"] == (["plain"] if variant == "distilled"
                                else ["", "plain"])
        tol = m["tolerances"]
        if size == GATE_SIZE:
            assert i["bf16"]
            by = r["bf16_vs_fp32_by_step"]
            assert tol["by_step_rel_l2"] == [2 * g for g in by]
            assert tol["latents_rel_l2"] == pytest.approx(
                2 * by[-1], rel=1e-12)
            assert "exact_rel_l2" not in tol
        else:
            assert set(tol) == {"exact_rel_l2", "rule"}
            assert tol["exact_rel_l2"] == EXACT_REL_L2
            read = m["results"]["gmlx_bf16"]
            keys = {str(k) for k in _kept(m)}
            assert set(read["rel_l2"]) == set(
                read["share_of_512_gate"]) == keys
    if not seen:
        pytest.skip("no check 6 manifest of FLUX.3 klein is committed")


@pytest.mark.parametrize("variant,size", CASES)
def test_the_seed_gives_the_noise_of_the_oracle(variant, size):
    from gmlx.image.samplers import noise_from_seed

    m = _manifest(variant, size)
    arrays, _ = fixture(m["name"])
    noise = arrays["noise"]
    assert noise.shape == (1, 128, size // 16, size // 16)
    assert np.array_equal(noise_from_seed(m["inputs"]["seed"], noise.shape),
                          noise)


@pytest.mark.parametrize("variant,size", CASES)
def test_the_schedule_and_times_are_the_reference_ones(variant, size):
    """The schedule of a request is the one generate ran bit for bit, and
    the time the DiT reads at each step is its entry in bf16."""
    from gmlx.image.api import ImagePipeline
    from gmlx.image.families.flux3_klein.model import _bf16
    from gmlx.image.registry import get_family

    m = _manifest(variant, size)
    arrays, _ = fixture(m["name"])
    pipe = ImagePipeline(dit=None, family=get_family("flux3_klein"),
                         variant=variant)
    steps = pipe.defaults.steps
    assert steps == m["inputs"]["steps"]
    got = pipe.sigmas(steps, (size // 16) ** 2)
    assert _bits(got) == _bits(arrays["schedule"])
    assert _bits(_bf16(got[:-1])) == _bits(arrays["times"])


@pytest.mark.integration
@pytest.mark.parametrize("variant", list(NAMES))
def test_check6_latents_at_512(pipes, variant):
    m = _manifest(variant, GATE_SIZE)
    arrays, _ = fixture(m["name"])
    gates = _gates(variant)
    got, finite = _run(pipes(m), arrays, m, variant, GATE_SIZE)
    print(_line(f"flux3 check6 {variant} {GATE_SIZE}", got, gates))
    assert finite
    assert sorted(got) == sorted(gates)
    over = {k: (v, gates[k]) for k, v in got.items() if v > gates[k]}
    assert not over, over


@pytest.mark.integration
@pytest.mark.parametrize("variant", list(NAMES))
def test_check6_bf16_at_1024_stays_finite(pipes, variant):
    """The run as it ships at 1024, read against the 512 gates."""
    m = _manifest(variant, EXACT_SIZE)
    arrays, _ = fixture(m["name"])
    gates = _gates(variant)
    got, finite = _run(pipes(m), arrays, m, variant, EXACT_SIZE)
    print(_line(f"flux3 check6 bf16 {variant} {EXACT_SIZE} against the "
                f"{GATE_SIZE} gates", got, gates))
    assert sorted(got) == sorted(gates)
    assert finite


def _time_inner_unrounded(mp, pipe):
    """The time embedding of 1000 t without the bf16 rounding of t."""
    from gmlx.image import ops
    from gmlx.image.families.flux3_klein import model as fm

    def embed(t, dim):
        t1000 = fm._bf16(np.float32(1000) * np.reshape(t, (-1,)).astype(
            np.float32))
        emb = ops.timestep_embedding(t1000, dim)
        return emb.astype(mx.bfloat16).astype(mx.float32)

    mp.setattr(fm, "time_embedding", embed)


def _own_time(mp, pipe):
    """A flow map that goes to its own time instead of the next."""
    from gmlx.image import api

    real = api.denoise_step

    def own_time(pipe, latents, sigma, conds, *, patch_table=None,
                 target=None):
        return real(pipe, latents, sigma, conds, patch_table=patch_table,
                    target=sigma)

    mp.setattr(api, "denoise_step", own_time)


def _rope_scale_one(mp, pipe):
    """The image positions at rope scale 1 instead of the family's."""
    mp.setattr(pipe.dit, "cfg",
               dataclasses.replace(pipe.dit.cfg, image_rope_scale=1.0))


# Wrong runs the exact gate must fail: the fault, the steps it runs (all
# with None) and whether the unconditional row is the prompt's.
EXACT_FAULTS = {
    "distilled": {
        "time embedding without its inner rounding":
            (_time_inner_unrounded, None, False),
        "flow map to its own time": (_own_time, 4, False),
        "image positions at rope scale 1": (_rope_scale_one, 2, False)},
    "base": {"prompt as the unconditional row": (None, 5, True)}}


@pytest.mark.integration
@pytest.mark.parametrize("variant", list(NAMES))
def test_check6_exact_at_1024(pipes, request, monkeypatch, variant):
    """The run in the oracle's fp32 arithmetic within EXACT_REL_L2 of its
    latents after each kept step and after the last, with every K-quant
    product dense, and each wrong run of EXACT_FAULTS over the gate at a
    kept step."""
    from gmlx.image import api
    from gmlx.image.conditioning import ByteLRU

    m = _manifest(variant, EXACT_SIZE)
    arrays, _ = fixture(m["name"])
    if not exact_process():
        pipes.release()
        return rerun_exact(request, "flux3 check6 exact")
    pipe = pipes(m)
    pipe.dit.act_dtype = mx.float32
    n = fg.exact_floats(pipe.dit, pipes.path)
    gate = {k: EXACT_REL_L2 for k in _kept(m)}
    with fg._exact(pipe.dit, arrays):
        got, finite = _run(pipe, arrays, m, variant, EXACT_SIZE)
    print(_line(f"flux3 check6 exact {variant} {EXACT_SIZE}, {n} float "
                "tensors in fp32", got, gate))
    assert finite and sorted(got) == sorted(gate)
    over = {k: v for k, v in got.items() if v > EXACT_REL_L2}
    for name, (apply, stop, neg) in EXACT_FAULTS[variant].items():
        with monkeypatch.context() as mp:
            # The memos of the clean run would hand the fault its
            # modulations and its caption.
            mp.setattr(pipe, "modulations", ByteLRU(api.MODULATION_BYTES))
            mp.setattr(pipe, "captions", ByteLRU(api.CAPTION_BYTES))
            if apply is not None:
                apply(mp, pipe)
            with fg._exact(pipe.dit, arrays):
                bad, _ = _run(pipe, arrays, m, variant, EXACT_SIZE,
                              stop=stop, neg=neg)
        print(_line(f"flux3 check6 exact {variant} {EXACT_SIZE}, {name}",
                    bad, gate))
        assert bad and max(bad.values()) > EXACT_REL_L2, (name, bad)
    assert not over, over


@pytest.mark.integration
def test_a_flow_map_to_its_own_time_fails_the_kept_steps(pipes,
                                                         monkeypatch):
    variant = "distilled"
    m = _manifest(variant, GATE_SIZE)
    arrays, _ = fixture(m["name"])
    _own_time(monkeypatch, None)
    gates = _gates(variant)
    got, _ = _run(pipes(m), arrays, m, variant, GATE_SIZE, stop=2)
    share = {k: v / gates[k] for k, v in got.items()}
    print(f"flux3 check6 own time: {share}")
    assert share and min(share.values()) > 1.0, share


@pytest.mark.integration
def test_base_without_its_unconditional_row_fails_the_kept_steps(pipes):
    variant = "base"
    m = _manifest(variant, GATE_SIZE)
    arrays, _ = fixture(m["name"])
    gates = _gates(variant)
    got, _ = _run(pipes(m), arrays, m, variant, GATE_SIZE, stop=2, neg=True)
    share = {k: v / gates[k] for k, v in got.items()}
    print(f"flux3 check6 prompt as the unconditional row: {share}")
    assert share and min(share.values()) > 1.0, share
