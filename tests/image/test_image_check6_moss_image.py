"""Check 6 of the image acceptance gate for MOSS-Image, on latents: a
whole guided sampling run on injected noise at the release's defaults,
50 steps at guidance 3 with block 22 left out of the negative pass. gmlx
samples from the seed's noise on the oracle's features of the plain
prompt and of the empty one, which is the release's negative prompt. The
two have different lengths, and each pass runs at its own. The seed's
noise must equal the oracle's, and the schedule must be the release's
bit for bit.

At 512 the run is the model as it ships, in bf16, and the latents after
each step the oracle kept and after the last must stay within twice the
oracle's own bf16 against fp32 gap after that step. The release's bf16
run drifts from its fp32 run about 2000 times over 50 steps, so these
gates guard the bf16 numerics alone: a negative pass that runs every
block stays inside them. The exact gate guards the skip set.

At 1024 the gate is the exact run: check 3's exact arithmetic, held to
the oracle's fp32 latents after each kept step and after the last at the
larger of EXACT_REL_L2 and twice gmlx's own fp32 floor after that step,
as check 3 holds its free rows. The run grows a difference about 3e4
times over its 50 steps, so float32 rounding alone reaches about 1e-3 by
the last step. The floor is the exact run from the noise moved one
float32 ulp against the exact run, which the 1024 manifest records and
the test measures again. A negative pass that runs every block, a positive pass that
leaves out block 22 too, the prompt as the negative, and the schedule
without its snap to the training grid each fail it. The bf16 run at 1024
must stay finite, and its readings print as a share of the 512 gate of
each step.

Skipped unless the manifests of scripts/image_oracle/check6_moss_image.py
are committed, GMLX_IMAGE_FIXTURES holds their arrays and
KQUANT_TEST_GGUF_DIR the DiT GGUF the oracle read."""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import block_gates as bg
import moss_gates as mg
import numpy as np
import pytest
from block_gates import exact_process, rerun_exact
from image_fixtures import image_dits
from oracle_fixture import ORACLE, committed, fixture, fx

NAME = "check6_moss_image"
VARIANT = "sft"
GATE_SIZE = 512
EXACT_SIZE = 1024
SIZES = (GATE_SIZE, EXACT_SIZE)
EXACT_REL_L2 = 1e-3


def _name(size: int) -> str:
    return f"{NAME}_{size}"


def _have(size: int) -> bool:
    return (ORACLE / f"{_name(size)}.manifest.json").is_file()


def _manifest(size: int) -> dict:
    if not _have(size):
        pytest.skip(f"{_name(size)}.manifest.json is not committed: run "
                    "scripts/image_oracle/check6_moss_image.py on the "
                    "MOSS-Image GGUF and commit the manifest")
    return committed(_name(size))


def _kept(m: dict) -> list:
    return m["inputs"]["kept_steps"] + [m["inputs"]["steps_run"]]


def _gates() -> dict:
    """The bf16 gate of the latents after each kept step and after the
    last, at the gate size."""
    m = _manifest(GATE_SIZE)
    tol = m["tolerances"]
    out = {k: tol[f"step{k}_rel_l2"] for k in m["inputs"]["kept_steps"]}
    out[m["inputs"]["steps_run"]] = tol["latents_rel_l2"]
    return out


def _ulps(got, want) -> int:
    g = np.asarray(got, np.float32).view(np.int32).astype(np.int64)
    w = np.asarray(want, np.float32).view(np.int32).astype(np.int64)
    return int(np.max(np.abs(g - w)))


class _Pipe:
    """The pipeline of the fixtures' file, loaded once."""

    def __init__(self, root):
        self.root, self.path, self.pipe = root, None, None

    def release(self) -> None:
        self.pipe = self.path = None
        mx.clear_cache()

    def __call__(self, m: dict):
        from gmlx.image import api

        if self.pipe is not None:
            return self.pipe
        want = m["files"]["dit"]
        for p in image_dits(self.root, "moss_image"):
            if Path(p).name == want["name"] and \
                    os.path.getsize(p) == want["size"]:
                break
        else:
            pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
        if fx.file_identity(p)["sha256"] != want["sha256"]:
            pytest.fail(f"{p} is not the file the oracle read")
        self.pipe = api.load(p, log=lambda *_: None)
        assert self.pipe.variant == VARIANT
        self.path = p
        return self.pipe


@pytest.fixture(scope="module")
def pipe(gguf_dir):
    return _Pipe(gguf_dir)


def _run(pipe, arrays: dict, m: dict, stop: int | None = None,
         noise=None, kept: dict | None = None, **kw) -> tuple[dict, bool]:
    """The relative L2 of the latents after each kept step and after the
    last against the oracle's fp32 latents, and whether the latents
    stayed finite after every step. ``stop`` ends the run after that
    many steps, ``noise`` replaces the fixture's, and ``kept`` takes the
    latents of the kept steps. ``kw`` goes to api.sample."""
    from gmlx.image import api
    from gmlx.image.conditioning import Conditioning

    size = m["inputs"]["size"]
    want = {k: arrays[f"latents_step{k}"] for k in m["inputs"]["kept_steps"]}
    want[m["inputs"]["steps_run"]] = arrays["latents"]
    cond = Conditioning(mx.array(arrays["features_positive"]),
                        f"check6 moss {size} plain")
    neg = Conditioning(mx.array(arrays["features_negative"]),
                       f"check6 moss {size} empty")
    kw.setdefault("neg", neg)
    sigmas = kw.pop("sigmas", None)
    if sigmas is None and stop is not None:
        sigmas = pipe.sigmas(m["inputs"]["steps"])
    if stop is not None:
        sigmas = sigmas[:stop + 1]
    got: dict = {}
    finite = [True]

    def on_step(i, x):
        a = np.array(x)
        finite[0] = finite[0] and bool(np.isfinite(a).all())
        if i + 1 in want:
            got[i + 1] = fx.rel_l2(a, want[i + 1])
            if kept is not None:
                kept[i + 1] = a

    api.sample(pipe, cond, height=size, width=size,
               noise=mx.array(arrays["noise"] if noise is None else noise),
               sigmas=sigmas,
               on_step=on_step, **kw)
    return got, finite[0]


def _line(label: str, got: dict, gates: dict) -> str:
    return f"{label}: " + ", ".join(
        f"step {k} {v:.2e} ({v / gates[k]:.2f} of its gate)"
        for k, v in sorted(got.items()))


def test_the_manifests_follow_the_rule():
    """The committed manifests alone: each run is the release's call at
    the variant's sampling row, the two prompts have different lengths,
    every gate at the gate size is twice the gap beside it, and each
    exact gate at the exact size follows the floor beside it."""
    from gmlx.image.registry import get_family

    fam = get_family("moss_image")
    row = fam.sampling[VARIANT]
    seen = 0
    for size in SIZES:
        if not _have(size):
            continue
        seen += 1
        m = committed(_name(size))
        i, r, tol = m["inputs"], m["results"], m["tolerances"]
        assert i["size"] == size
        assert i["steps"] == row.steps
        assert i["guidance_scale"] == row.cfg
        assert tuple(i["uncond_skip_layers"]) == row.neg_skip
        assert i["scheduler"]["shift"] == row.shift
        assert i["negative_prompt"] == ""
        assert i["blocks"] == 42
        rows = i["text_rows"]
        assert rows["positive"] != rows["negative"]
        assert (i["positive_key"], i["negative_key"]) == (
            "features_plain", "features_empty")
        assert m["files"]["dit"]["name"] == "MOSS-Image-SFT-Q4_K_M.gguf"
        if size == GATE_SIZE:
            by = r["bf16_vs_fp32_by_step"]
            assert len(by) == i["steps_run"]
            for k in i["kept_steps"]:
                assert tol[f"step{k}_rel_l2"] == 2 * by[k - 1], k
            assert tol["latents_rel_l2"] == pytest.approx(
                2 * by[-1], rel=1e-12)
            assert "exact_rel_l2" not in tol
        else:
            assert set(tol) == {"exact_rel_l2", "rule", "exact_by_step"}
            assert tol["exact_rel_l2"] == EXACT_REL_L2
            floor = r["gmlx_exact_floor"]
            assert set(floor) == {str(k) for k in _kept(m)}
            assert tol["exact_by_step"] == {
                k: max(EXACT_REL_L2, 2 * v) for k, v in floor.items()}
    if not seen:
        pytest.skip("no check 6 manifest of MOSS-Image is committed")


@pytest.mark.parametrize("size", SIZES)
def test_the_seed_gives_the_noise_of_the_oracle(size):
    from gmlx.image.samplers import noise_from_seed

    m = _manifest(size)
    arrays, _ = fixture(m["name"])
    noise = arrays["noise"]
    assert noise.shape == (1, 128, size // 16, size // 16)
    assert np.array_equal(noise_from_seed(m["inputs"]["seed"], noise.shape),
                          noise)


@pytest.mark.parametrize("size", SIZES)
def test_the_schedule_is_the_release_one(size):
    """The schedule of a request is the release's bit for bit, and the
    release's timesteps are its sigmas times 1000 in float32, as the DiT
    forms them."""
    from gmlx.image.api import ImagePipeline
    from gmlx.image.registry import get_family

    m = _manifest(size)
    arrays, _ = fixture(m["name"])
    pipe = ImagePipeline(dit=None, family=get_family("moss_image"),
                         variant=VARIANT)
    steps = pipe.defaults.steps
    assert steps == m["inputs"]["steps"]
    got, want = pipe.sigmas(steps, (size // 16) ** 2), arrays["sigmas"]
    assert got.shape == want.shape
    assert want.size - 1 == m["inputs"]["steps_run"]
    assert _ulps(got, want) == 0
    ts = want[:-1] * np.float32(1000)
    assert _ulps(ts, arrays["timesteps"]) == 0


@pytest.mark.integration
def test_check6_latents_at_512(pipe):
    m = _manifest(GATE_SIZE)
    arrays, _ = fixture(m["name"])
    gates = _gates()
    got, finite = _run(pipe(m), arrays, m)
    print(_line(f"moss check6 {GATE_SIZE}", got, gates))
    assert finite
    assert sorted(got) == sorted(gates)
    over = {k: (v, gates[k]) for k, v in got.items() if v > gates[k]}
    assert not over, over


@pytest.mark.integration
def test_check6_bf16_at_1024_stays_finite(pipe):
    """The run as it ships at 1024, read against the 512 gates."""
    m = _manifest(EXACT_SIZE)
    arrays, _ = fixture(m["name"])
    gates = _gates()
    got, finite = _run(pipe(m), arrays, m)
    print(_line(f"moss check6 bf16 {EXACT_SIZE} against the {GATE_SIZE} "
                "gates", got, gates))
    assert sorted(got) == sorted(gates)
    assert finite


def _skip_positive_too(mp):
    """Block 22 left out of the positive pass as well."""
    from gmlx.image import api

    real = api.denoise_step

    def both(pipe, latents, sigma, conds, *, skip=(), **kw):
        return real(pipe, latents, sigma, conds, skip=skip or (22,), **kw)

    mp.setattr(api, "denoise_step", both)


def _unsnapped(steps: int) -> np.ndarray:
    from gmlx.image import schedule

    v = schedule.linspace_f32(1.0, 5.0 / 1004, steps)
    return np.append(schedule.shift_static(v, 5.0), np.float32(0))


def _floor(p, arrays: dict, m: dict, clean: dict) -> dict:
    """The exact run from the noise moved one float32 ulp against the
    exact run ``clean``, at each kept step."""
    up = np.nextafter(arrays["noise"], np.float32(np.inf))
    moved: dict = {}
    with mg._exact(p.dit):
        _run(p, arrays, m, noise=up, kept=moved)
    return {k: fx.rel_l2(moved[k], clean[k]) for k in sorted(clean)}


def _exact_gate(m: dict, now: dict) -> dict:
    """The gate of each kept step from the recorded floor, after checking
    that the floor measured now gives a gate within 1.25 times of it."""
    rec = m["results"].get("gmlx_exact_floor")
    shown = ", ".join(f"step {k} {v!r}" for k, v in now.items())
    print(f"moss check6 exact {EXACT_SIZE}, fp32 floor: {shown}")
    assert rec is not None, f"{m['name']} records no gmlx_exact_floor"
    gate = {int(k): v for k, v in m["tolerances"]["exact_by_step"].items()}
    assert sorted(gate) == sorted(now)
    for k, v in now.items():
        assert max(EXACT_REL_L2, 2 * v) <= 1.25 * gate[k], (k, v, gate[k])
    return gate


# Wrong runs the exact gate must fail: the fault, the steps it runs (all
# with None) and the arguments it gives api.sample.
EXACT_FAULTS = {
    "every block in the negative pass": (None, 10, {"neg_skip": ()}),
    "block 22 left out of the positive pass too":
        (_skip_positive_too, 10, {}),
    "the prompt as the negative": (None, 10, {"neg": "cond"}),
    "the schedule not snapped": (None, None, {"sigmas": "unsnapped"}),
}


@pytest.mark.integration
def test_check6_exact_at_1024(pipe, request, monkeypatch):
    """The run in the oracle's fp32 arithmetic within its gate of the
    oracle's latents after each kept step and after the last, with every
    K-quant product dense, and each wrong run of EXACT_FAULTS over the
    gate at a kept step."""
    from gmlx.image import api
    from gmlx.image.conditioning import ByteLRU, Conditioning

    m = _manifest(EXACT_SIZE)
    arrays, _ = fixture(m["name"])
    if not exact_process():
        pipe.release()
        return rerun_exact(request, "moss check6 exact")
    p = pipe(m)
    n = mg.exact_floats(p.dit, pipe.path)
    clean: dict = {}
    with mg._exact(p.dit):
        got, finite = _run(p, arrays, m, kept=clean)
    gate = _exact_gate(m, _floor(p, arrays, m, clean))
    print(_line(f"moss check6 exact {EXACT_SIZE}, {n} float tensors in "
                "fp32", got, gate))
    assert finite and sorted(got) == sorted(gate)
    over = {k: (v, gate[k]) for k, v in got.items() if v > gate[k]}
    for name, (apply, stop, kw) in EXACT_FAULTS.items():
        kw = dict(kw)
        if kw.get("neg") == "cond":
            kw["neg"] = Conditioning(mx.array(arrays["features_positive"]),
                                     f"check6 moss {EXACT_SIZE} plain neg")
        if isinstance(kw.get("sigmas"), str):
            kw["sigmas"] = _unsnapped(m["inputs"]["steps"])
        with monkeypatch.context() as mp:
            # The memos of the clean run would hand the fault its
            # modulations and its caption.
            mp.setattr(p, "modulations", ByteLRU(api.MODULATION_BYTES))
            mp.setattr(p, "captions", ByteLRU(api.CAPTION_BYTES))
            if apply is not None:
                apply(mp)
            # A fault may leave a block out of both passes, so the check
            # that every product ran dense is the clean run's alone.
            with bg.exact_products(p.dit) as used:
                bad, _ = _run(p, arrays, m, stop=stop, **kw)
            assert not used.kernel, (name, used.kernel[:3])
        print(_line(f"moss check6 exact {EXACT_SIZE}, {name}", bad, gate))
        assert bad and any(v > gate[k] for k, v in bad.items()), (name, bad)
    assert not over, over
