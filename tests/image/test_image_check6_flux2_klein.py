"""Check 6 of the image acceptance gate for FLUX.2 klein, on latents: a
whole sampling run on injected noise, for the distilled checkpoint and
for Base, at 4B and at 9B. gmlx samples from the seed's noise with the
variant's defaults on the oracle's conditioning, the empty prompt's for
the unconditional row of Base. The seed's noise must equal the oracle's,
and the schedule and the times the DiT saw must be the reference's.

At 512 the run is the model as it ships, in bf16, and the latents after
each step the oracle kept and after the last must stay within twice the
oracle's own bf16 against fp32 gap after that step. A distilled run on
the schedule of another image size, and a Base run whose unconditional
row is the prompt's, fail the kept steps.

At 1024 the gate is the exact run: check 3's exact arithmetic, held to
EXACT_REL_L2 of the oracle's fp32 latents after each kept step and after
the last. The time embedding without its inner bf16 rounding, the
schedule of another image size, and Base with the prompt as its
unconditional row each fail it. The bf16 run at 1024 must stay finite,
and its readings print as a share of the 512 gate of each step, as the
manifest records them. The wrong runs at 512 are of the 4B files, those
of the exact run at 1024 of the 4B and the 9B files, and the 9B cases
declare a Mac of 24 GB.

Skipped unless the manifests of scripts/image_oracle/check6_flux2_klein.py
are committed, GMLX_IMAGE_FIXTURES holds their arrays and
KQUANT_TEST_GGUF_DIR the DiT GGUFs the oracle read."""
from __future__ import annotations

import os
from pathlib import Path

import flux2_gates as fg
import mlx.core as mx
import numpy as np
import pytest
from block_gates import exact_process, rerun_exact
from image_fixtures import image_dits
from oracle_fixture import ORACLE, committed, fixture, fx

# A Mac of 16 GB, as most users run (tests/host_box.py).
pytestmark = pytest.mark.host_gb(16)

# A kind is a variant, with "9b " before it for the 9B files.
NAMES = {"distilled": "check6_flux2_klein", "base": "check6_flux2_klein_base",
         "9b distilled": "check6_flux2_klein_9b",
         "9b base": "check6_flux2_klein_9b_base"}
DITS = {"distilled": "flux-2-klein-4b-Q4_K_M.gguf",
        "base": "flux-2-klein-base-4b-Q4_K_M.gguf",
        "9b distilled": "flux-2-klein-9b-Q4_K_M.gguf",
        "9b base": "flux-2-klein-base-9b-Q4_K_M.gguf"}
MODEL_INFO = {"distilled": "flux.2-klein-4b", "base": "flux.2-klein-base-4b",
              "9b distilled": "flux.2-klein-9b",
              "9b base": "flux.2-klein-base-9b"}
GATE_SIZE = 512
EXACT_SIZE = 1024
SIZES = (GATE_SIZE, EXACT_SIZE)
EXACT_REL_L2 = 1e-3
REFERENCE_COMMIT = "50fe5162777813d869182b139e83b10743caef15"


def variant_of(kind: str) -> str:
    return kind.removeprefix("9b ")


def family_of(kind: str) -> str:
    return "flux2_klein_9b" if kind.startswith("9b ") else "flux2_klein"


def _param(kind: str, *rest):
    """A case of ``kind``, the 9B ones on a Mac of 24 GB."""
    marks = [pytest.mark.host_gb(24)] if kind.startswith("9b ") else []
    return pytest.param(kind, *rest, marks=marks)


KINDS = [_param(k) for k in NAMES]
CASES = [_param(k, s) for k in NAMES for s in SIZES]


def _name(kind: str, size: int) -> str:
    return f"{NAMES[kind]}_{size}"


def _have(kind: str, size: int) -> bool:
    return (ORACLE / f"{_name(kind, size)}.manifest.json").is_file()


def _manifest(kind: str, size: int) -> dict:
    if not _have(kind, size):
        pytest.skip(f"{_name(kind, size)}.manifest.json is not "
                    "committed: run scripts/image_oracle/check6_flux2_klein.py"
                    " on the FLUX.2 klein GGUF and commit the manifest")
    return committed(_name(kind, size))


def _bits(a) -> list:
    return np.asarray(a, np.float32).view(np.uint32).tolist()


def _kept(m: dict) -> list:
    return m["inputs"]["kept_steps"] + [m["inputs"]["steps"]]


def _gates(kind: str) -> dict:
    """The bf16 gate of the latents after each kept step and after the
    last, at the gate size."""
    m = _manifest(kind, GATE_SIZE)
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
        found = [p for fam in ("flux2_klein", "flux2_klein_9b")
                 for p in image_dits(self.root, fam)]
        for p in found:
            if Path(p).name == want["name"] and \
                    os.path.getsize(p) == want["size"]:
                break
        else:
            pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
        if fx.file_identity(p)["sha256"] != want["sha256"]:
            pytest.fail(f"{p} is not the file the oracle read")
        self.pipe = api.load(p, log=lambda *_: None)
        self.path = p
        assert self.pipe.variant == m["inputs"]["variant"]
        assert self.pipe.family.name == m["family"]
        return self.pipe


@pytest.fixture(scope="module")
def pipes(gguf_dir):
    return _Pipes(gguf_dir)


def _conds(arrays: dict, variant: str, size: int):
    from gmlx.image.conditioning import Conditioning

    cond = Conditioning(mx.array(arrays["features"]),
                        f"check6 flux2 {variant} {size}")
    neg = None
    if variant == "base":
        neg = Conditioning(mx.array(arrays["features_empty"]),
                           f"check6 flux2 {variant} {size} empty")
    return cond, neg


def _run(pipe, arrays: dict, m: dict, kind: str, size: int,
         stop: int | None = None, neg=False,
         tokens: int | None = None) -> tuple[dict, bool]:
    """The relative L2 of the latents after each kept step and after the
    last against the oracle's fp32 latents, and whether the latents
    stayed finite after every step. ``stop`` ends the run after that
    many steps, and ``tokens`` takes the schedule of that token count."""
    from gmlx.image import api

    steps = m["inputs"]["steps"]
    want = {k: arrays[f"latents_step{k}"] for k in m["inputs"]["kept_steps"]}
    want[steps] = arrays["latents"]
    cond, uncond = _conds(arrays, variant_of(kind), size)
    if neg:
        uncond = cond
    got: dict = {}
    finite = [True]

    def on_step(i, x):
        a = np.array(x)
        finite[0] = finite[0] and bool(np.isfinite(a).all())
        if i + 1 in want:
            got[i + 1] = fx.rel_l2(a, want[i + 1])

    sigmas = pipe.sigmas(steps, tokens or (size // 16) ** 2)
    if stop is not None:
        sigmas = sigmas[:stop + 1]
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
    commit at FLUX2_MODEL_INFO's steps and guidance for the file's size
    and variant, every gate at the gate size is twice the gap beside it,
    and the exact size holds the exact tolerance and the bf16 readings of
    each kept step."""
    seen = [_follows_the_rule(k, s) for k in NAMES for s in SIZES
            if _have(k, s)]
    if not seen:
        pytest.skip("no check 6 manifest of FLUX.2 klein is committed")


def _follows_the_rule(kind: str, size: int) -> None:
    from gmlx.image.registry import get_family

    variant = variant_of(kind)
    fam = get_family(family_of(kind))
    m = committed(_name(kind, size))
    assert m["family"] == fam.name
    i, r = m["inputs"], m["results"]["run"]
    row = fam.sampling[variant]
    assert m["files"]["dit"]["name"] == DITS[kind]
    assert i["reference_commit"] == REFERENCE_COMMIT
    assert (i["variant"], i["size"], i["refs"]) == (variant, size, 0)
    assert i["model_info"] == MODEL_INFO[kind]
    assert (i["steps"], i["guidance"]) == (row.steps, row.cfg)
    assert i["tokens"] == (size // 16) ** 2
    # denoise_cfg runs the empty prompt's row beside the prompt's.
    assert r["rows"] == ([1] if variant == "distilled" else [2])
    tol = m["tolerances"]
    if size == GATE_SIZE:
        assert i["bf16"]
        by = r["bf16_vs_fp32_by_step"]
        assert tol["by_step_rel_l2"] == [2 * g for g in by]
        assert tol["latents_rel_l2"] == pytest.approx(2 * by[-1], rel=1e-12)
        assert "exact_rel_l2" not in tol
    else:
        assert set(tol) == {"exact_rel_l2", "rule"}
        assert tol["exact_rel_l2"] == EXACT_REL_L2
        read = m["results"]["gmlx_bf16"]
        keys = {str(k) for k in _kept(m)}
        assert set(read["rel_l2"]) == set(read["share_of_512_gate"]) == keys


@pytest.mark.parametrize("kind,size", CASES)
def test_the_seed_gives_the_noise_of_the_oracle(kind, size):
    from gmlx.image.samplers import noise_from_seed

    m = _manifest(kind, size)
    arrays, _ = fixture(m["name"])
    noise = arrays["noise"]
    assert noise.shape == (1, 128, size // 16, size // 16)
    assert np.array_equal(noise_from_seed(m["inputs"]["seed"], noise.shape),
                          noise)


@pytest.mark.parametrize("kind,size", CASES)
def test_the_schedule_and_times_are_the_reference_ones(kind, size):
    """The schedule of a request is the one the reference ran bit for
    bit, and the time the DiT reads at each step is its entry in bf16."""
    from gmlx.image.api import ImagePipeline
    from gmlx.image.families.flux2_klein.model import _bf16
    from gmlx.image.registry import get_family

    m = _manifest(kind, size)
    arrays, _ = fixture(m["name"])
    pipe = ImagePipeline(dit=None, family=get_family(family_of(kind)),
                         variant=variant_of(kind))
    steps = pipe.defaults.steps
    assert steps == m["inputs"]["steps"]
    got = pipe.sigmas(steps, (size // 16) ** 2)
    assert _bits(got) == _bits(arrays["schedule"])
    assert _bits(_bf16(got[:-1])) == _bits(arrays["times"])


@pytest.mark.integration
@pytest.mark.parametrize("kind", KINDS)
def test_check6_latents_at_512(pipes, kind):
    m = _manifest(kind, GATE_SIZE)
    arrays, _ = fixture(m["name"])
    gates = _gates(kind)
    got, finite = _run(pipes(m), arrays, m, kind, GATE_SIZE)
    print(_line(f"flux2 check6 {kind} {GATE_SIZE}", got, gates))
    assert finite
    assert sorted(got) == sorted(gates)
    over = {k: (v, gates[k]) for k, v in got.items() if v > gates[k]}
    assert not over, over


@pytest.mark.integration
@pytest.mark.parametrize("kind", KINDS)
def test_check6_bf16_at_1024_stays_finite(pipes, kind):
    """The run as it ships at 1024, read against the 512 gates."""
    m = _manifest(kind, EXACT_SIZE)
    arrays, _ = fixture(m["name"])
    gates = _gates(kind)
    got, finite = _run(pipes(m), arrays, m, kind, EXACT_SIZE)
    print(_line(f"flux2 check6 bf16 {kind} {EXACT_SIZE} against the "
                f"{GATE_SIZE} gates", got, gates))
    assert sorted(got) == sorted(gates)
    assert finite


def _time_inner_unrounded(mp, pipe):
    """The time embedding of 1000 t without the bf16 rounding of t."""
    from gmlx.image import ops
    from gmlx.image.families.flux2_klein import model as fm

    def embed(t, dim):
        t1000 = fm._bf16(np.float32(1000) * np.reshape(t, (-1,)).astype(
            np.float32))
        emb = ops.timestep_embedding(t1000, dim)
        return emb.astype(mx.bfloat16).astype(mx.float32)

    mp.setattr(fm, "time_embedding", embed)


# The schedule of a 512 image, where the run is at 1024.
OTHER_TOKENS = 1024

# Wrong runs the exact gate must fail, by variant, at 4B and 9B, whose
# schedules and times at a size are the same: the fault, the steps it
# runs (all with None), whether the unconditional row is the prompt's,
# and the token count of the schedule (the image's with None).
EXACT_FAULTS = {
    "distilled": {
        "time embedding without its inner rounding":
            (_time_inner_unrounded, None, False, None),
        "the schedule of a 512 image": (None, 2, False, OTHER_TOKENS)},
    # Without its inner rounding the time embedding changes only at the
    # steps whose 1000 times the time rounds to another bf16 value: the
    # last of the distilled schedule at 1024, and steps 7, 15, 16, 19, 21,
    # 22 and 23 of Base's before its kept step 25.
    "base": {
        "time embedding without its inner rounding":
            (_time_inner_unrounded, 25, False, None),
        "prompt as the unconditional row": (None, 5, True, None)}}


@pytest.mark.integration
@pytest.mark.parametrize("kind", KINDS)
def test_check6_exact_at_1024(pipes, request, monkeypatch, kind):
    """The run in the oracle's fp32 arithmetic within EXACT_REL_L2 of its
    latents after each kept step and after the last, with every K-quant
    product dense, and each wrong run of EXACT_FAULTS over the gate at a
    kept step."""
    from gmlx.image import api
    from gmlx.image.conditioning import ByteLRU

    m = _manifest(kind, EXACT_SIZE)
    arrays, _ = fixture(m["name"])
    if not exact_process():
        pipes.release()
        return rerun_exact(request, "flux2 check6 exact")
    pipe = pipes(m)
    pipe.dit.act_dtype = mx.float32
    n = fg.exact_floats(pipe.dit, pipes.path)
    gate = {k: EXACT_REL_L2 for k in _kept(m)}
    with fg._exact(pipe.dit):
        got, finite = _run(pipe, arrays, m, kind, EXACT_SIZE)
    print(_line(f"flux2 check6 exact {kind} {EXACT_SIZE}, {n} float "
                "tensors in fp32", got, gate))
    assert finite and sorted(got) == sorted(gate)
    over = {k: v for k, v in got.items() if v > EXACT_REL_L2}
    faults = EXACT_FAULTS[variant_of(kind)]
    for name, (apply, stop, neg, tokens) in faults.items():
        with monkeypatch.context() as mp:
            # The memos of the clean run would hand the fault its
            # modulations and its caption.
            mp.setattr(pipe, "modulations", ByteLRU(api.MODULATION_BYTES))
            mp.setattr(pipe, "captions", ByteLRU(api.CAPTION_BYTES))
            if apply is not None:
                apply(mp, pipe)
            with fg._exact(pipe.dit):
                bad, _ = _run(pipe, arrays, m, kind, EXACT_SIZE,
                              stop=stop, neg=neg, tokens=tokens)
        print(_line(f"flux2 check6 exact {kind} {EXACT_SIZE}, {name}",
                    bad, gate))
        assert bad and max(bad.values()) > EXACT_REL_L2, (name, bad)
    assert not over, over


@pytest.mark.integration
def test_another_image_size_schedule_fails_the_kept_steps(pipes):
    """A distilled run at 512 on the schedule of a 1024 image."""
    variant = "distilled"
    m = _manifest(variant, GATE_SIZE)
    arrays, _ = fixture(m["name"])
    gates = _gates(variant)
    got, _ = _run(pipes(m), arrays, m, variant, GATE_SIZE, stop=2,
                  tokens=4096)
    share = {k: v / gates[k] for k, v in got.items()}
    print(f"flux2 check6 the schedule of a 1024 image: {share}")
    assert share and min(share.values()) > 1.0, share


@pytest.mark.integration
def test_base_without_its_unconditional_row_fails_the_kept_steps(pipes):
    variant = "base"
    m = _manifest(variant, GATE_SIZE)
    arrays, _ = fixture(m["name"])
    gates = _gates(variant)
    got, _ = _run(pipes(m), arrays, m, variant, GATE_SIZE, stop=2, neg=True)
    share = {k: v / gates[k] for k, v in got.items()}
    print(f"flux2 check6 prompt as the unconditional row: {share}")
    assert share and min(share.values()) > 1.0, share
