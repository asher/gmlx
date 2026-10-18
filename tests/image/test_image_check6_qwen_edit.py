"""Check 6 of the image acceptance gate for Qwen-Image 2.1 with reference
images: a guided edit, on the real files.

The oracle is the reference pipeline's own call for a prompt, its
reference images and a negative prompt at a guidance above 1. Each case
of the fixture runs three times here:

  in the oracle's arithmetic   the first two steps of ``sample`` on the
      oracle's float32 features, image rows and reference latents, with
      fp32 activations and exact products in a child process. The
      latents after each step are held to the free-running exact gate of
      block_gates.py. This is the row that holds the layout of the
      references in the prefix: the two references in the other order
      must miss it.
  from the oracle's conditioning   the same inputs on the path a request
      takes, all the steps: the DiT and the guided loop. The final
      latents are held to the gate of the case, twice the oracle's own
      bf16 against fp32 gap, and the latents after each step the oracle
      kept to twice that gap after that step.
  from the files   ``gmlx image generate`` with ``--ref``, from the source
      images and the prompts: the resize, the encoder with its vision
      tower, the VAE encode of the references and the same loop. The
      final latents are held to the gate of the case plus the gate of a
      reference encode (below). For the case with one reference the
      image is held to 30 dB of the float32 decode of the oracle's
      latents, both by gmlx's VAE, as the image of a prompt is.

The reference latents of gmlx's VAE are held to the oracle's by the
relative L2 gate that check 5 sets for a bf16 encode at 512 x 512. A run
from the files starts from those latents where the oracle's bf16 run
starts from its own float32 ones, and an edit that keeps its picture
carries the difference to its output one to one: with one reference the
final latents move by the 2.3e-02 that the reference latents differ by.
So the gate of a run from the files is the sum of the two gates.

The two cases differ in kind. The edit of one picture stays on its
reference and a bf16 loop drifts little: the oracle's own bf16 run ends
6.7e-03 from its fp32 run. The composition of two pictures is free and
its bf16 run ends 1.2e-01 away. In the oracle's arithmetic gmlx matches
the oracle to 1.1e-06 after the first step of that case and to 6.0e-03
after the last, so its final latents hold little beside the exact row.

Skipped unless KQUANT_TEST_GGUF_DIR holds the DiT, the Qwen3-VL-8B text
GGUF and its mmproj that the fixtures name, GMLX_TEST_IMAGE_DIR the
Qwen-Image 2.1 VAE, and GMLX_IMAGE_FIXTURES the fixture.
scripts/image_oracle/check6_qwen_image21_edit.py writes it."""
from __future__ import annotations

import contextlib
import gc
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from image_dir import vae_files
from image_fixtures import image_dits
from oracle_fixture import ORACLE, committed, fixture, fx

import mlx.core as mx

from block_gates import (QWEN_EXACT_FREE_TOL, exact_process, exact_products,
                         rerun_exact)
from test_image_check2_qwen_edit import _find

# Each row loads local model files, so it runs on this Mac
# (tests/host_box.py).
pytestmark = pytest.mark.integration

NAME = "check6_qwen_image21_edit"
CHECK2 = "check2_qwen_image21_edit"
CHECK5 = "check5_qwen_image21_vae"
FAMILY = "qwen_image21"
CASES = ("one", "two")
PSNR_DB = 30.0


@pytest.fixture(scope="module")
def check6():
    if not (ORACLE / f"{NAME}.manifest.json").is_file():
        pytest.skip(f"{NAME}.manifest.json is not committed: run "
                    "scripts/image_oracle/check6_qwen_image21_edit.py and "
                    "commit the manifest it writes")
    return fixture(NAME)


@pytest.fixture(scope="module")
def files(check6, gguf_dir, gguf_index):
    from gmlx.image import text_encoder as te

    want = check6[1]["files"]["dit"]
    dits = [p for p in image_dits(gguf_dir, FAMILY)
            if Path(p).name == want["name"]
            and os.path.getsize(p) == want["size"]]
    spec = te.encoder_spec(FAMILY)
    enc_files = committed(CHECK2)["files"]
    enc = _find(gguf_index.get(spec.arch, []), enc_files["encoder"])
    mmproj = _find(
        [str(p) for p in gguf_dir.rglob(enc_files["mmproj"]["name"])],
        enc_files["mmproj"])
    vaes = sorted(vae_files(FAMILY), key=lambda p: p.suffix != ".gguf")
    if not (dits and enc and mmproj and vaes):
        pytest.skip("needs the Qwen-Image 2.1 DiT of the fixture, the "
                    "Qwen3-VL-8B text GGUF and its mmproj under "
                    "KQUANT_TEST_GGUF_DIR, and the Qwen-Image 2.1 VAE under "
                    "GMLX_TEST_IMAGE_DIR")
    if fx.file_identity(dits[0])["sha256"] != want["sha256"]:
        pytest.fail(f"{dits[0]} is not the file the oracle read")
    return dits[0], enc, mmproj, str(vaes[0])


def _conds(arrays, m, case: str):
    """The oracle's two conditionings of a case."""
    from gmlx.image.conditioning import Conditioning

    row = m["inputs"]["cases"][case]
    keys = [row[k].split(" ")[0].removeprefix("tap_")
            for k in ("features", "negative_features")]
    refs = tuple(mx.array(arrays[f"ref_latents_{n}"]) for n in row["images"])
    return [Conditioning(mx.array(arrays[f"features_{k}"]),
                         f"check6 edit {k}", ref_latents=refs,
                         image_slots=tuple(int(i)
                                           for i in arrays[f"slots_{k}"]))
            for k in keys]


def _gate(m: dict, case: str) -> tuple[float, str]:
    tol = m["tolerances"]["cases"][case]
    return tol["latents_rel_l2"], (
        f" (the gap of case {tol['from']})" if tol.get("from") else "")


def _files_gate(m: dict, case: str) -> float:
    """The gate of the final latents of a run from the files: the gate
    of the case plus the gate of a reference encode."""
    return _gate(m, case)[0] + committed(CHECK5)["tolerances"]["sizes"][
        "512x512"]["encode_bf16_rel_l2"]


def _step_gates(m: dict, case: str) -> dict:
    """The gate of the latents after each kept step, by the manifest's
    rule: twice the oracle's own bf16 against fp32 gap after that step,
    of the case or of the case its final gate comes from."""
    src = m["tolerances"]["cases"][case].get("from", case)
    by_step = m["results"]["cases"][src]["bf16_vs_fp32_by_step"]
    return {k: 2 * by_step[k - 1] for k in m["inputs"]["kept_steps"]}


def test_the_inputs_of_the_oracle(check6, files):
    """The noise is the seed's, the schedule is the pipeline's at the
    target's token count, whatever the references, and each case names
    features that the check 2 fixture with images holds."""
    from gmlx.image import api
    from gmlx.image.samplers import noise_from_seed

    arrays, m = check6
    ins = m["inputs"]
    side = ins["size"] // 16
    assert arrays["noise"].shape == (1, 64, side, side)
    assert np.array_equal(noise_from_seed(ins["seed"], arrays["noise"].shape),
                          arrays["noise"])
    assert ins["guidance"] > 1.0 and ins["kv_cache"] is True
    fam = api.sampler_defaults(FAMILY, "base")
    assert fam.steps == ins["steps"]
    m2 = committed(CHECK2)
    for case in CASES:
        row = ins["cases"][case]
        assert row["negative_prompt"] and row["prompt"]
        for kind in ("features", "negative_features"):
            key = row[kind].split(" ")[0]
            assert arrays[f"features_{key.removeprefix('tap_')}"].shape == \
                tuple(m2["tensors"][key]["shape"])
        for n, shape in zip(row["images"], row["reference_latents"]):
            assert list(arrays[f"ref_latents_{n}"].shape) == shape
            assert shape[1] * 16 == arrays[f"resized_{n}"].shape[0]
    assert len(ins["cases"]["one"]["images"]) == 1
    assert len(ins["cases"]["two"]["images"]) == 2


@pytest.fixture(scope="module")
def pipe(files, generated):
    """The DiT and the VAE, loaded after the runs of the command have
    returned their models."""
    from gmlx.image import api

    dit, _, _, vae = files
    return api.load(dit, vae=vae, log=lambda *_: None)


def test_the_schedule_of_an_edit_is_the_schedule_of_the_pipeline(check6,
                                                                 pipe):
    arrays, m = check6
    ins = m["inputs"]
    got = pipe.sigmas(ins["steps"], (ins["size"] // 16) ** 2)

    def bits(a):
        return np.asarray(a, np.float32).view(np.uint32).tolist()

    assert bits(got) == bits(arrays["sigmas"])
    assert bits(got[:-1] * np.float32(1000)) == bits(arrays["timesteps"])


@pytest.mark.parametrize("name", ["a", "b", "c"])
def test_reference_latents_of_the_vae_against_the_oracle(check6, pipe, name):
    """The VAE encode of a resized reference, alpha and all, against the
    reference pipeline's float32 encode."""
    from gmlx.image import api
    from gmlx.image.conditioning import Reference

    arrays, m = check6
    if f"ref_latents_{name}" not in arrays:
        pytest.skip(f"no case of the fixture holds image {name}")
    tol = committed(CHECK5)["tolerances"]["sizes"]["512x512"][
        "encode_bf16_rel_l2"]
    ref = Reference(arrays[f"resized_{name}"])
    got = np.array(api._reference_latents(pipe, ref, None))
    want = arrays[f"ref_latents_{name}"]
    assert got.shape == want.shape
    r = fx.compare(got, want)
    print(f"qwen edit reference latents {name} "
          f"{ref.width}x{ref.height}: rel L2 {r['rel_l2']:.4e}, "
          f"{r['rel_l2'] / tol:.2f} of the gate {tol:.4e}, cosine "
          f"{r['cosine']:.6f}")
    assert r["rel_l2"] <= tol, r


@pytest.mark.parametrize("case", CASES)
def test_check6_edit_final_latents_from_the_conditioning_of_the_oracle(
        check6, pipe, case):
    from gmlx.image import api

    arrays, m = check6
    ins = m["inputs"]
    cond, neg = _conds(arrays, m, case)
    kept: dict = {}

    def on_step(i, x):
        if i + 1 in ins["kept_steps"]:
            kept[i + 1] = fx.rel_l2(np.array(x),
                                    arrays[f"latents_{case}_step{i + 1}"])

    x = api.sample(pipe, cond, neg=neg, cfg=ins["guidance"],
                   height=ins["size"], width=ins["size"],
                   noise=arrays["noise"], on_step=on_step)
    got = fx.compare(np.array(x), arrays[f"latents_{case}"])
    tol, note = _gate(m, case)
    gates = _step_gates(m, case)
    print(f"qwen check6 edit {case}, the oracle's conditioning: rel L2 "
          f"{got['rel_l2']:.4e}, {got['rel_l2'] / tol:.2f} of its gate "
          f"{tol:.4e}{note}, cosine {got['cosine']:.6f}; after steps "
          + ", ".join(f"{k} {v:.2e} ({v / gates[k]:.2f} of its gate)"
                      for k, v in sorted(kept.items())))
    assert sorted(kept) == ins["kept_steps"]
    assert got["rel_l2"] <= tol, got
    over = {k: (v, gates[k]) for k, v in kept.items() if v > gates[k]}
    assert not over, over
    # Two prefix states, one a conditioning, each with its references.
    assert len(pipe.captions) >= 2


@pytest.fixture(scope="module")
def exact_pipe(files):
    """The DiT alone, for the child process of the exact row."""
    from gmlx.image import api

    return api.load(files[0], log=lambda *_: None)


EXACT_STEPS = 2


@pytest.mark.parametrize("case", CASES)
def test_check6_edit_first_steps_in_the_arithmetic_of_the_oracle(
        check6, files, request, case):
    """The layout of the references and the guided step, at the precision
    of the oracle's arithmetic: the latents after each of the first steps
    are within the free-running exact gate, and with two references in
    the other order they are not."""
    from gmlx.image import api

    arrays, m = check6
    ins = m["inputs"]
    assert set(range(1, EXACT_STEPS + 1)) <= set(ins["kept_steps"])
    if not exact_process():
        return rerun_exact(request, "qwen check6 edit exact")
    pipe = request.getfixturevalue("exact_pipe")

    def run(cond, neg) -> dict:
        kept: dict = {}

        def on_step(i, x):
            kept[i + 1] = fx.rel_l2(
                np.array(x), arrays[f"latents_{case}_step{i + 1}"])

        with exact_products(pipe.dit) as used:
            api.sample(pipe, cond, neg=neg, cfg=ins["guidance"],
                       height=ins["size"], width=ins["size"],
                       noise=arrays["noise"],
                       sigmas=arrays["sigmas"][:EXACT_STEPS + 1],
                       on_step=on_step)
        assert not used.kernel, used.kernel[:3]
        return kept

    kept = run(*_conds(arrays, m, case))
    print(f"qwen check6 edit exact {case}: after steps "
          + ", ".join(f"{k} {v:.2e}" for k, v in sorted(kept.items()))
          + f", gate {QWEN_EXACT_FREE_TOL:.0e}")
    assert sorted(kept) == list(range(1, EXACT_STEPS + 1))
    assert all(v <= QWEN_EXACT_FREE_TOL for v in kept.values()), kept
    refs = [arrays[f"ref_latents_{n}"] for n in ins["cases"][case]["images"]]
    if len(refs) < 2:
        return
    assert refs[0].shape != refs[1].shape
    try:
        wrong = run(*_swapped(arrays, m, case))
    except ValueError as e:
        # The layout refuses latents of another shape than their slots.
        print(f"qwen check6 edit exact {case}, references swapped: refused, "
              f"{e}")
        return
    print(f"qwen check6 edit exact {case}, references swapped: after steps "
          + ", ".join(f"{k} {v:.2e} ({v / QWEN_EXACT_FREE_TOL:.0f} gates)"
                      for k, v in sorted(wrong.items())))
    assert all(v > QWEN_EXACT_FREE_TOL for v in wrong.values()), wrong


def _swapped(arrays, m, case):
    cond, neg = _conds(arrays, m, case)
    from gmlx.image.conditioning import Conditioning

    return [Conditioning(c.features, c.key + " swapped",
                         ref_latents=c.ref_latents[::-1],
                         image_slots=c.image_slots) for c in (cond, neg)]


def test_check6_edit_fails_with_the_references_in_the_other_order(check6,
                                                                  pipe):
    """The must-fail of the layout on the path a request takes: the
    reference latents of the two images in the other order miss the gate
    of the first kept step."""
    from gmlx.image import api

    arrays, m = check6
    ins = m["inputs"]
    refs = [arrays[f"ref_latents_{n}"] for n in ins["cases"]["two"]["images"]]
    if refs[0].shape == refs[1].shape:
        pytest.skip("the two references have one shape")
    cond, neg = _swapped(arrays, m, "two")
    first = min(ins["kept_steps"])
    kept: dict = {}

    class Stop(Exception):
        pass

    def on_step(i, x):
        if i + 1 == first:
            kept[first] = fx.rel_l2(np.array(x),
                                    arrays[f"latents_two_step{first}"])
            raise Stop

    with pytest.raises((Stop, ValueError)) as e:
        api.sample(pipe, cond, neg=neg, cfg=ins["guidance"],
                   height=ins["size"], width=ins["size"],
                   noise=arrays["noise"], on_step=on_step)
    if e.type is not Stop:
        # The layout refuses latents of another shape than their slots.
        print(f"qwen check6 edit two, references swapped: refused, {e.value}")
        return
    gate = _step_gates(m, "two")[first]
    print(f"qwen check6 edit two, references swapped: rel L2 "
          f"{kept[first]:.4e} after step {first}, "
          f"{kept[first] / gate:.1f} of its gate {gate:.4e}")
    assert kept[first] > gate


@pytest.fixture(scope="module")
def generated(check6, files, tmp_path_factory):
    """One ``gmlx image generate --ref`` run per case, from the source
    images and the prompts: the saved latents, the PNG pixels and the
    JSON record."""
    from PIL import Image

    from gmlx.commands.cli import umbrella_main
    from gmlx.image import api

    arrays, m = check6
    ins = m["inputs"]
    dit, enc, mmproj, vae = files
    d = tmp_path_factory.mktemp("qwen_edit")
    out = {}
    encode = api.encode_text
    conds: list = []

    def kept(pipe, prompt, **kw):
        cond = encode(pipe, prompt, **kw)
        conds.append((prompt, [int(i) for i in cond.image_slots],
                      np.array(cond.features.astype(mx.float32))))
        return cond

    for case in CASES:
        conds.clear()
        row = ins["cases"][case]
        refs = []
        for n in row["images"]:
            Image.fromarray(arrays[f"source_{n}"]).save(d / f"{n}.png")
            refs += ["--ref", str(d / f"{n}.png")]
        lat, png = d / f"{case}.npz", d / f"{case}.png"
        buf = io.StringIO()
        api.encode_text = kept
        try:
            with contextlib.redirect_stdout(buf):
                rc = umbrella_main([
                    "image", "generate", dit, "--encoder", enc,
                    "--encoder-mmproj", mmproj, "--vae", vae, "--prompt",
                    row["prompt"], "--negative-prompt",
                    row["negative_prompt"], "--guidance",
                    str(ins["guidance"]), *refs, "--ref-size",
                    str(ins["reference_resolution"]), "--size",
                    f"{ins['size']}x{ins['size']}", "--seed",
                    str(ins["seed"]), "--output", str(png), "--save-latent",
                    str(lat), "--json"])
        finally:
            api.encode_text = encode
        assert rc == 0, buf.getvalue()
        with Image.open(png) as im:
            pixels = np.asarray(im.convert("RGB"))
        out[case] = {"latents": np.load(lat)["latents"], "pixels": pixels,
                     "record": json.loads(
                         buf.getvalue().strip().splitlines()[-1]),
                     "conds": list(conds)}
        gc.collect()
        mx.clear_cache()
    return out


@pytest.mark.parametrize("case", CASES)
def test_check6_edit_final_latents_from_the_files(check6, generated, case):
    arrays, m = check6
    ins = m["inputs"]
    rec = generated[case]["record"]
    s = rec["settings"]
    assert rec["model"]["family"] == FAMILY and "mmproj" in rec["model"]
    assert (s["steps"], s["guidance"]) == (ins["steps"], ins["guidance"])
    assert s["negative_prompt"] == ins["cases"][case]["negative_prompt"]
    names = ins["cases"][case]["images"]
    assert [r["size"] for r in s["refs"]] == [
        "{1}x{0}".format(*arrays[f"resized_{n}"].shape[:2]) for n in names]
    got = fx.compare(generated[case]["latents"], arrays[f"latents_{case}"])
    loop, note = _gate(m, case)
    tol = _files_gate(m, case)
    print(f"qwen check6 edit {case}, from the files: rel L2 "
          f"{got['rel_l2']:.4e}, {got['rel_l2'] / tol:.2f} of its gate "
          f"{tol:.4e}, the gate of the loop {loop:.4e}{note} plus the gate "
          f"of a reference encode; cosine {got['cosine']:.6f}")
    assert got["rel_l2"] <= tol, got


@pytest.mark.parametrize("case", CASES)
def test_check6_edit_conditionings_from_the_files(check6, generated, case):
    """The command encodes the prompt and the negative prompt each with
    the references, as the reference pipeline does: each conditioning of
    a run from the files has the oracle's image rows, and its features are
    within the check 2 gate of the oracle's."""
    arrays, m = check6
    row = m["inputs"]["cases"][case]
    keys = [row[k].split(" ")[0].removeprefix("tap_")
            for k in ("features", "negative_features")]
    gates = committed(CHECK2)["tolerances"]["rel_l2"]
    got = generated[case]["conds"]
    assert [p for p, _, _ in got] == [row["prompt"], row["negative_prompt"]]
    for (_, slots, feats), k in zip(got, keys):
        assert slots == arrays[f"slots_{k}"].tolist(), k
        rel = fx.rel_l2(feats, arrays[f"features_{k}"])
        print(f"qwen check6 edit {k}, the command's conditioning: rel L2 "
              f"{rel:.4e}, {rel / gates[k]:.2f} of the check 2 gate")
        assert rel <= gates[k], k


def test_the_image_of_an_edit(check6, files, generated):
    """The PNG of the edit with one reference against the float32 decode
    of the oracle's latents, by gmlx's VAE."""
    from gmlx.image.vae import load_vae, tiling

    arrays, m = check6
    size = m["inputs"]["size"]
    fp32 = load_vae(files[3], family=FAMILY, dtype=mx.float32)
    ref = tiling.decode(fp32, mx.array(arrays["latents_one"]), tiled=False)
    ref = np.clip(np.array(ref) / 2 + 0.5, 0.0, 1.0)[0, ..., :3]
    del fp32
    pixels = generated["one"]["pixels"]
    assert pixels.shape == (size, size, 3)
    db = fx.psnr(pixels.astype(np.float64) / 255.0, ref, 1.0)
    print(f"qwen edit image {size}: the written PNG {db:.2f} dB against "
          f"the float32 decode of the oracle's latents, bar {PSNR_DB:.0f}")
    assert db >= PSNR_DB
