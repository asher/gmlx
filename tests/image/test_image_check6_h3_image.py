"""Check 6 of the image acceptance gate for MiniMax-H3-Image, on latents: a
whole text to image run on injected noise at the release's settings, the
six steps of the PDD grid at guidance 1. gmlx samples from the seed's
noise on the oracle's features of the plain prompt, check 2's fp32 tap.
The seed's noise must equal the oracle's, and the schedule must be the
release's grid bit for bit.

At 1024x1024 the run is the model as it ships, in bf16 with its kernel
routes, and the latents after each step must stay within twice the
oracle's own bf16 against fp32 gap after that step.

At 2048x2048 and 4096x4096 the gate is the exact run: check 3's exact
arithmetic, every K-quant product dense on its dequantized weight and
every float tensor in float32, held to the oracle's fp32 latents after
each step at the larger of EXACT_REL_L2 and twice gmlx's own fp32 floor
after that step. The floor is the largest relative L2 of the exact run
against the same run from the noise and features nudged by one float32
step, four seeds, as video H3's check 6 floors its run but for its CPU
stream run. The manifest records it (GMLX_IMAGE_RECORD_FLOORS=<dir>
writes manifests with the floors measured now), and the test measures it
again, within 1.25 times. Each step from the oracle's latents before it
must give the oracle's velocity within EXACT_TOL. A head one grid point
off, head k without head 0, and each step's target one grid point late
each fail the gate at 2048. The bf16 run at 2048 and 4096 must stay
finite, and its readings print as a share of the 1024 gate of each step.

An edit at 1024x1024 with two pictures runs at the same gate: gmlx's
copies of each picture must be ComfyUI's byte for byte, its VAE encodes
each VAE copy on the oracle's draws (ComfyUI's seed 42 torch draws, as
Reference noise) within half the share of the oracle's latents those
draws make (other draws land about 1.4 times that share away; the
encode's own numerics are check 5's), and the latents after each step,
from those references, the
oracle's features and the noise of each reference's rows, must stay
within twice the oracle's own bf16 against fp32 gap after that step. The
run without its references and the run with its vision rows read as text
each miss a gate. The run on the family's own seed 42 draws prints as a
share of the gates.

Skipped unless the manifests of scripts/image_oracle/check6_h3_image.py
are committed, GMLX_IMAGE_FIXTURES holds their arrays and
KQUANT_TEST_GGUF_DIR the DiT GGUF and the VAE the oracle read."""
from __future__ import annotations

import dataclasses
import json
import os
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import block_gates as bg
import h3_image_gates as hg
from block_gates import exact_process, rerun_exact
from image_fixtures import image_dits
from oracle_fixture import ORACLE, committed, fixture, fx

from gmlx.image.families.h3_image import edit as he
from gmlx.image.families.h3_image import family as hi
from gmlx.image.families.h3_image import model as him

NAME = "check6_h3_image"
VARIANT = "pruned"
GATE_SIZE = 1024
EXACT_SIZES = (2048, 4096)
FAULT_SIZE = 2048
SIZES = (GATE_SIZE, *EXACT_SIZES)
EXACT_REL_L2 = 1e-3
EDIT = f"{NAME}_edit_{GATE_SIZE}"


def _name(size: int) -> str:
    return f"{NAME}_{size}"


def _have(size: int) -> bool:
    return (ORACLE / f"{_name(size)}.manifest.json").is_file()


def _manifest(size: int) -> dict:
    if not _have(size):
        pytest.skip(f"{_name(size)}.manifest.json is not committed: run "
                    "scripts/image_oracle/check6_h3_image.py on the "
                    "MiniMax-H3-Image GGUF and commit the manifest")
    return committed(_name(size))


def _edit_manifest() -> dict:
    if not (ORACLE / f"{EDIT}.manifest.json").is_file():
        pytest.skip(f"{EDIT}.manifest.json is not committed: run "
                    "scripts/image_oracle/check6_h3_image.py --edit and "
                    "commit the manifest")
    return committed(EDIT)


def _gates(m: dict | None = None) -> dict:
    """The bf16 gate of the latents after each step, at the gate size, of
    ``m``, the text to image run by default."""
    m = _manifest(GATE_SIZE) if m is None else m
    return {k: m["tolerances"][f"step{k}_rel_l2"]
            for k in range(1, m["inputs"]["steps"] + 1)}


def _ulps(got, want) -> int:
    g = np.asarray(got, np.float32).view(np.int32).astype(np.int64)
    w = np.asarray(want, np.float32).view(np.int32).astype(np.int64)
    return int(np.max(np.abs(g - w)))


def _find(root: Path, want: dict) -> str:
    for p in sorted(Path(root).rglob(want["name"])):
        if p.is_file() and os.path.getsize(p) == want["size"]:
            if fx.file_identity(str(p))["sha256"] != want["sha256"]:
                pytest.fail(f"{p} is not the file the oracle read")
            return str(p)
    pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")


class _Pipe:
    """The pipeline of the fixtures' file, loaded once, with the VAE of a
    manifest that names one."""

    def __init__(self, root):
        self.root, self.path, self.pipe = root, None, None

    def release(self) -> None:
        self.pipe = self.path = None
        mx.clear_cache()

    def __call__(self, m: dict):
        from gmlx.image import api

        vae = m["files"].get("vae")
        if self.pipe is not None and (vae is None
                                      or self.pipe.vae is not None):
            return self.pipe
        self.release()
        want = m["files"]["dit"]
        for p in image_dits(self.root, "h3_image"):
            if Path(p).name == want["name"] and \
                    os.path.getsize(p) == want["size"]:
                break
        else:
            pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
        if fx.file_identity(p)["sha256"] != want["sha256"]:
            pytest.fail(f"{p} is not the file the oracle read")
        self.pipe = api.load(p, vae=None if vae is None
                             else _find(self.root, vae),
                             log=lambda *_: None)
        assert self.pipe.variant == VARIANT
        self.path = p
        return self.pipe


@pytest.fixture(scope="module")
def pipe(gguf_dir):
    return _Pipe(gguf_dir)


def _run(pipe, arrays: dict, m: dict, noise=None, features=None,
         kept: dict | None = None, cond=None) -> tuple[dict, bool]:
    """The relative L2 of the latents after each step against the oracle's
    fp32 latents, and whether they stayed finite after every step.
    ``noise`` and ``features`` replace the fixture's, ``cond`` the
    conditioning of its features, and ``kept`` takes the latents."""
    from gmlx.image import api
    from gmlx.image.conditioning import Conditioning

    size = m["inputs"]["size"]
    steps = m["inputs"]["steps"]
    want = {k: arrays[f"latents_step{k}"] for k in range(1, steps + 1)}
    if cond is None:
        cond = Conditioning(mx.array(arrays["features"] if features is None
                                     else features),
                            f"check6 h3 image {size} plain")
    got: dict = {}
    finite = [True]

    def on_step(i, x):
        a = np.array(x)
        finite[0] = finite[0] and bool(np.isfinite(a).all())
        got[i + 1] = fx.rel_l2(a, want[i + 1])
        if kept is not None:
            kept[i + 1] = a

    api.sample(pipe, cond, height=size, width=size,
               noise=mx.array(arrays["noise"] if noise is None else noise),
               on_step=on_step)
    return got, finite[0]


def _line(label: str, got: dict, gates: dict) -> str:
    return f"{label}: " + ", ".join(
        f"step {k} {v:.2e} ({v / gates[k]:.2f} of its gate)"
        for k, v in sorted(got.items()))


def test_the_manifests_follow_the_rule():
    """The committed manifests alone: each run is the release's six steps
    at guidance 1 on the plain prompt, every gate at the gate size is
    twice the gap beside it, and each exact gate at the other sizes
    follows the floor beside it."""
    row = hi.FAMILY.sampling[VARIANT]
    seen = 0
    for size in SIZES:
        if not _have(size):
            continue
        seen += 1
        m = committed(_name(size))
        i, r, tol = m["inputs"], m["results"], m["tolerances"]
        assert i["size"] == size and i["gate_size"] == GATE_SIZE
        assert i["steps"] == r["steps_run"] == row.steps == len(row.sigmas)
        assert i["guidance"] == row.cfg == 1.0
        assert i["prompt"] == "plain"
        assert m["files"]["dit"]["name"] == hi.VARIANTS[next(iter(
            hi.VARIANTS))].files[0]
        if size == GATE_SIZE:
            by = r["bf16_vs_fp32_by_step"]
            assert len(by) == i["steps"]
            for k in range(1, i["steps"] + 1):
                assert tol[f"step{k}_rel_l2"] == 2 * by[k - 1], k
            assert tol["latents_rel_l2"] == tol[f"step{i['steps']}_rel_l2"]
            assert "exact_rel_l2" not in tol
        else:
            assert tol["exact_rel_l2"] == EXACT_REL_L2
            assert "latents_rel_l2" not in tol
            floor = r.get("gmlx_exact_floor")
            if floor is not None:
                assert set(floor) == {str(k) for k in
                                      range(1, i["steps"] + 1)}
                assert tol["exact_by_step"] == {
                    k: max(EXACT_REL_L2, 2 * v["rel_l2"])
                    for k, v in floor.items()}
    if not seen:
        pytest.skip("no check 6 manifest of MiniMax-H3-Image is committed")


@pytest.mark.parametrize("size", SIZES)
def test_the_seed_gives_the_noise_of_the_oracle(size):
    from gmlx.image.samplers import noise_from_seed

    m = _manifest(size)
    arrays, _ = fixture(m["name"])
    noise = arrays["noise"]
    assert noise.shape == (1, 24, size // 16, size // 16)
    assert np.array_equal(noise_from_seed(m["inputs"]["seed"], noise.shape),
                          noise)


@pytest.mark.parametrize("size", SIZES)
def test_the_schedule_is_the_release_grid(size):
    """The schedule of a request is the PDD grid bit for bit, and the
    oracle's timesteps are its sigmas times 1000 in float32, as the DiT
    forms them."""
    from gmlx.image.api import ImagePipeline

    m = _manifest(size)
    arrays, _ = fixture(m["name"])
    pipe = ImagePipeline(dit=None, family=hi.FAMILY, variant=VARIANT)
    got = pipe.sigmas(pipe.defaults.steps)
    want = arrays["sigmas"]
    assert got.shape == want.shape and want.size - 1 == m["inputs"]["steps"]
    assert _ulps(got, want) == 0
    assert _ulps(want[:-1] * np.float32(1000), arrays["timesteps"]) == 0


@pytest.mark.integration
def test_check6_latents_at_1024(pipe):
    m = _manifest(GATE_SIZE)
    arrays, _ = fixture(m["name"])
    gates = _gates()
    got, finite = _run(pipe(m), arrays, m)
    print(_line(f"h3 image check6 {GATE_SIZE}", got, gates))
    assert finite
    assert sorted(got) == sorted(gates)
    over = {k: (v, gates[k]) for k, v in got.items() if v > gates[k]}
    assert not over, over


@pytest.mark.integration
@pytest.mark.parametrize("size", EXACT_SIZES)
def test_check6_bf16_stays_finite(pipe, size):
    """The run as it ships, read against the 1024 gates."""
    m = _manifest(size)
    arrays, _ = fixture(m["name"])
    gates = _gates()
    got, finite = _run(pipe(m), arrays, m)
    print(_line(f"h3 image check6 bf16 {size} against the {GATE_SIZE} "
                "gates", got, gates))
    assert sorted(got) == sorted(gates)
    assert finite


def _head_one_off(mp):
    real = hi.grid_point
    mp.setattr(hi, "grid_point",
               lambda s: min(real(s) + 1, len(hi.PDD_SIGMAS) - 1))


def _head_alone(mp):
    """Head k without head 0, as a slice of the bank reads it."""
    real = him.H3ImageDiT.head

    def head(self, x, mod, plan, taps=None):
        out = real(self, x, mod, plan, taps)
        if not mod.heads:
            return out
        return out - real(self, x, dataclasses.replace(mod, heads=()), plan)
    mp.setattr(him.H3ImageDiT, "head", head)


def _target_one_late(mp):
    """Each step handed the grid point after the sigma it goes to as its
    target, so it reads the heads of two intervals. A step's own sigma as
    its target reads its own interval's head, as the right target does
    on the grid, and changes nothing."""
    from gmlx.image import api

    real = api.denoise_step
    grid = np.asarray(hi.PDD_SIGMAS, np.float32)

    def late(pipe, latents, sigma, conds, *, target=None, **kw):
        if target is not None:
            k = int(np.argmin(np.abs(grid - np.float32(target))))
            target = grid[min(k + 1, len(grid) - 1)]
        return real(pipe, latents, sigma, conds, target=target, **kw)
    mp.setattr(api, "denoise_step", late)


# Wrong runs the exact gate must fail at FAULT_SIZE.
EXACT_FAULTS = {"head one grid point off": _head_one_off,
                "head k without head 0": _head_alone,
                "each step's target one grid point late": _target_one_late}


@contextmanager
def _fresh_memos(p):
    """``p`` with empty modulation and caption memos, so a run forms both
    again and runs every product of the DiT."""
    from gmlx.image import api
    from gmlx.image.conditioning import ByteLRU

    saved = p.modulations, p.captions
    p.modulations = ByteLRU(api.MODULATION_BYTES)
    p.captions = ByteLRU(api.CAPTION_BYTES)
    try:
        yield
    finally:
        p.modulations, p.captions = saved


def _floor(p, arrays: dict, m: dict, clean: dict) -> dict:
    """Step to (relative L2, run) of the largest distance of the exact run
    ``clean`` from the same run from the noise and features nudged by one
    float32 step (hg.NUDGE_SEEDS), after the step. Video H3's check 6 also
    takes the run on the CPU stream: here that run takes 20 minutes at
    2048 and sets no gate."""
    runs: dict = {}
    for seed in hg.NUDGE_SEEDS:
        n = hg.nudged({"latents": arrays["noise"],
                       "features": arrays["features"]}, seed)
        runs[f"nudge {seed}"] = got = {}
        with _fresh_memos(p), hg._exact(p.dit):
            _run(p, arrays, m, noise=n["latents"], features=n["features"],
                 kept=got)
    out = {}
    for k in sorted(clean):
        r = {name: fx.rel_l2(v[k], clean[k]) for name, v in runs.items()}
        top = max(r, key=r.get)
        out[k] = (r[top], top)
    return out


def _forced(p, arrays: dict, m: dict) -> dict:
    """The relative L2 of each step's velocity from the oracle's latents
    before it against the oracle's velocity at that step."""
    from gmlx.image import api
    from gmlx.image.conditioning import Conditioning

    s = p.sigmas(m["inputs"]["steps"])
    cond = Conditioning(mx.array(arrays["features"]),
                        f"check6 h3 image {m['inputs']['size']} plain")
    out = {}
    for k in range(1, len(s)):
        x = arrays["noise"] if k == 1 else arrays[f"latents_step{k - 1}"]
        v = api.denoise_step(p, mx.array(x), s[k - 1], cond, target=s[k])
        out[k] = fx.rel_l2(np.array(v), arrays[f"velocity_step{k}"])
    return out


def _record(m: dict, now: dict) -> dict:
    """The manifest with the floors measured now and their gates, written
    to the GMLX_IMAGE_RECORD_FLOORS directory."""
    how = (f"the largest relative L2 between gmlx's exact run and the "
           f"same run from the noise and features nudged by one float32 "
           f"step ({len(hg.NUDGE_SEEDS)} seeds), after the step; "
           f"tests/image/test_image_check6_h3_image.py _floor on "
           f"{m['files']['dit']['name']} "
           f"{m['files']['dit']['sha256'][:8]}, "
           f"{mx.device_info()['device_name']}")
    floor = {str(k): {"rel_l2": v, "run": run, "measured": how}
             for k, (v, run) in now.items()}
    gate = {str(k): max(EXACT_REL_L2, 2 * v) for k, (v, _) in now.items()}
    path = ORACLE / f"{m['name']}.manifest.json"
    doc = json.loads(path.read_text())
    assert doc["npz"] == m["npz"], path
    doc["results"]["gmlx_exact_floor"] = floor
    doc["tolerances"]["exact_by_step"] = gate
    dst = hg.record_dir() / path.name
    tmp = dst.with_name(dst.name + ".part")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(dst)
    return doc


def _exact_gate(m: dict, now: dict, label: str) -> dict:
    """The gate of each step from the recorded floor, after checking that
    the floor measured now is within 1.25 times of it."""
    print(f"{label}, fp32 floor: " + ", ".join(
        f"step {k} {v!r} ({run})" for k, (v, run) in now.items()))
    rec = m["results"].get("gmlx_exact_floor")
    assert rec is not None, (
        f"{m['name']} records no gmlx_exact_floor: run the exact gate with "
        f"{hg.RECORD}=<dir>")
    gate = {int(k): v for k, v in m["tolerances"]["exact_by_step"].items()}
    assert sorted(gate) == sorted(now)
    for k, (v, _) in now.items():
        r = rec[str(k)]["rel_l2"]
        assert r / 1.25 <= v <= 1.25 * r, (
            f"step {k} floor {v:.3e} is not within 1.25x of the recorded "
            f"{r:.3e}; on another machine, or after an mlx, mlx-kquant or "
            f"macOS update, run the exact gate with {hg.RECORD}=<dir>")
    return gate


@pytest.mark.integration
@pytest.mark.parametrize("size", EXACT_SIZES)
def test_check6_exact(pipe, request, monkeypatch, size):
    """The run in the oracle's fp32 arithmetic within its gate of the
    oracle's latents after each step, every K-quant product dense, each
    step from the oracle's latents within EXACT_TOL of its velocity, and
    at FAULT_SIZE each wrong run of EXACT_FAULTS over the gate at a
    step."""
    m = _manifest(size)
    arrays, _ = fixture(m["name"])
    record = hg.record_dir() is not None
    if not exact_process():
        pipe.release()
        return rerun_exact(request, "h3 image check6 exact")
    p = pipe(m)
    label = f"h3 image check6 exact {size}"
    clean: dict = {}
    with hg.exact_floats(p.dit) as n:
        with hg._exact(p.dit):
            got, finite = _run(p, arrays, m, kept=clean)
        with _fresh_memos(p), hg._exact(p.dit):
            forced = _forced(p, arrays, m)
        print(_line(f"{label}, each step from the oracle's latents, its "
                    f"velocity", forced, dict.fromkeys(forced, bg.EXACT_TOL)))
        now = _floor(p, arrays, m, clean)
        if record:
            m = _record(m, now)
        gate = _exact_gate(m, now, label)
        print(_line(f"{label}, {n} float tensors cast to fp32", got, gate))
        assert finite and sorted(got) == sorted(gate)
        over = {k: (v, gate[k]) for k, v in got.items() if v > gate[k]}
        if size == FAULT_SIZE:
            for name, apply in EXACT_FAULTS.items():
                with monkeypatch.context() as mp, _fresh_memos(p):
                    apply(mp)
                    with bg.exact_products(p.dit) as used:
                        bad, _ = _run(p, arrays, m)
                    assert not used.kernel, (name, used.kernel[:3])
                print(_line(f"{label}, {name}", bad, gate))
                assert any(v > gate[k] for k, v in bad.items()), (name, bad)
    assert not over, over
    late = {k: v for k, v in forced.items() if v > bg.EXACT_TOL}
    assert not late, late


def _edit_refs(arrays: dict, m: dict) -> tuple:
    """gmlx's references of the edit's pictures."""
    return he.prepare([arrays[f"picture_{i}"]
                       for i in range(1, len(m["inputs"]["refs"]) + 1)])


def test_the_edit_manifest_follows_the_rule():
    """The committed edit manifest alone: the release's six steps at
    guidance 1 at the gate size, picture 1's VAE copy the size of the run,
    and every gate twice the gap beside it."""
    m = _edit_manifest()
    i, r, tol = m["inputs"], m["results"], m["tolerances"]
    row = hi.FAMILY.sampling[VARIANT]
    assert i["edit"] and i["size"] == i["gate_size"] == GATE_SIZE
    assert i["steps"] == r["steps_run"] == row.steps
    assert i["guidance"] == row.cfg == 1.0
    assert i["refs"][0] == [GATE_SIZE, GATE_SIZE] and len(i["refs"]) >= 2
    assert m["files"]["dit"]["name"] == hi.VARIANTS[next(iter(
        hi.VARIANTS))].files[0]
    by = r["bf16_vs_fp32_by_step"]
    for k in range(1, i["steps"] + 1):
        assert tol[f"step{k}_rel_l2"] == 2 * by[k - 1], k
    assert "exact_rel_l2" not in tol


def test_the_edit_takes_comfyuis_copies_and_its_vision_spans():
    """gmlx's two copies of each picture are the oracle's, which are
    MiniMaxH3ImageConditioning's, byte for byte; the edit takes picture
    1's VAE size; and the oracle's token tags are 0 over a block of one
    row per 32 x 32 pixels of each vision copy and its two markers."""
    from gmlx.image.h3.pictures import spans_of

    m = _edit_manifest()
    arrays, _ = fixture(EDIT)
    refs = _edit_refs(arrays, m)
    for i, ref in enumerate(refs, 1):
        assert np.array_equal(ref.pixels[..., :3], arrays[f"vae_copy_{i}"])
        assert np.array_equal(ref.vision, arrays[f"vision_copy_{i}"])
    assert [list(he.default_size(refs))] == m["inputs"]["refs"][:1]
    spans = spans_of(arrays["text_tags"])
    assert [list(sp) for sp in spans] == m["inputs"]["spans"]
    assert [b - a for a, b in spans] == [
        (r.vision.shape[0] // 32) * (r.vision.shape[1] // 32) + 2
        for r in refs]


def _edit_cond(p, arrays: dict, m: dict, own_draws: bool = False):
    """The edit's conditioning: the oracle's features and image slots,
    gmlx's references of the pictures on the oracle's draws (the family's
    own with ``own_draws``), encoded by gmlx's VAE, and the noise of each
    reference's rows."""
    from gmlx.image import api
    from gmlx.image.conditioning import Conditioning
    from gmlx.image.h3.pictures import span_slots, spans_of

    refs = _edit_refs(arrays, m)
    if not own_draws:
        refs = tuple(dataclasses.replace(r, noise=arrays[
            f"ref_vae_noise_{i}"]) for i, r in enumerate(refs))
    noise = tuple(mx.array(arrays[f"ref_noise_{i}"])
                  for i in range(len(refs)))
    cond = Conditioning(mx.array(arrays["features"]),
                        f"check6 h3 image edit {own_draws}",
                        image_slots=span_slots(spans_of(arrays["text_tags"])),
                        refs=refs, ref_noise=noise)
    return api.encode_references(p, cond)


@pytest.mark.integration
def test_check6_edit_at_1024(pipe, monkeypatch):
    m = _edit_manifest()
    arrays, _ = fixture(EDIT)
    gates = _gates(m)
    p = pipe(m)
    cond = _edit_cond(p, arrays, m)
    share = m["results"]["edit"]["draw_share_of_encode"]
    enc = [fx.rel_l2(np.array(z), arrays[f"ref_latents_{i}"][0])
           for i, z in enumerate(cond.ref_latents)]
    print("h3 image check6 edit, each reference's latents: " + ", ".join(
        f"{v:.2e} ({v / s:.3f} of its draws' share)"
        for v, s in zip(enc, share)))
    got, finite = _run(p, arrays, m, cond=cond)
    print(_line(f"h3 image check6 edit {GATE_SIZE}", got, gates))
    assert finite and sorted(got) == sorted(gates)
    over = {k: (v, gates[k]) for k, v in got.items() if v > gates[k]}
    assert not over, over
    assert all(v <= s / 2 for v, s in zip(enc, share)), (enc, share)
    own, _ = _run(p, arrays, m, cond=_edit_cond(p, arrays, m, True))
    print(_line("h3 image check6 edit on the family's own seed 42 draws",
                own, gates))
    faults = {"without its references": dataclasses.replace(
                  cond, refs=(), ref_latents=(), ref_hashes=(),
                  ref_noise=()),
              "vision rows read as text": dataclasses.replace(
                  cond, image_slots=())}
    for name, bad in faults.items():
        miss, _ = _run(p, arrays, m, cond=bad)
        print(_line(f"h3 image check6 edit {name}", miss, gates))
        assert any(v > gates[k] for k, v in miss.items()), (name, miss)
