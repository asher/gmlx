"""Check 6 of the image acceptance gate for FLUX.3 klein edits: a whole
sampling run with one reference image and one with two, on injected
noise, against BFL's own generate.

The preparation of a reference, its canvas, pad or crop, the box that
crops the output and the default size, equal the reference's for every
source the oracle prepared, bit for bit. The ids of the reference tokens
are those the DiT of the reference read. The reference latents of gmlx's
VAE are held to the oracle's float32 encode by the gate check 5 sets for
an encode of that dtype at 512 pixels. From the oracle's conditioning,
its features and reference latents, the latents after each kept step and
after the last stay within twice the oracle's own bf16 against fp32 gap
after that step. The two references in the other order miss those
gates.

Skipped unless the manifest of scripts/image_oracle/check6_flux3_klein_edit.py
is committed and GMLX_IMAGE_FIXTURES holds its arrays, and, for the runs,
KQUANT_TEST_GGUF_DIR the DiT GGUF the oracle read and GMLX_TEST_IMAGE_DIR
the FLUX.2 VAE."""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from image_dir import vae_files
from image_fixtures import image_dits
from oracle_fixture import ORACLE, committed, fixture, fx

NAME = "check6_flux3_klein_edit"
FAMILY = "flux3_klein"
CHECK5 = "check5_flux3_klein_vae"
CASES = ["one", "two"]


@pytest.fixture(scope="module")
def check6():
    if not (ORACLE / f"{NAME}.manifest.json").is_file():
        pytest.skip(f"{NAME}.manifest.json is not committed: run "
                    "scripts/image_oracle/check6_flux3_klein_edit.py on the "
                    "FLUX.3 klein GGUF and commit the manifest")
    arrays, _ = fixture(NAME)
    return arrays, committed(NAME)


def _pixels(ref) -> np.ndarray:
    assert (ref.pixels[..., 3] == 255).all()
    return ref.pixels[..., :3]


def _sources(arrays, row) -> list:
    return [arrays[f"source_{k}"] for k in row["sources"]]


@pytest.mark.parametrize("rule", ["single", "multi"])
def test_the_references_are_prepared_as_the_reference_prepares_them(
        check6, rule):
    from gmlx.image.families.flux3_klein import edit

    arrays, m = check6
    rows = {k: r for k, r in m["inputs"]["prepared"].items()
            if k.split("_", 1)[0] == rule}
    assert len(rows) >= 4
    for key, row in rows.items():
        src = arrays[f"source_{key.split('_', 1)[1]}"]
        if rule == "single":
            got = edit.prepare([src])
            want_crop = None if row["crop"] is None else tuple(row["crop"])
            assert got.crop == want_crop, key
        else:
            got = (edit.prepare_multi_reference(src),)
        assert edit.default_size(got) == tuple(row["size"]), key
        assert np.array_equal(_pixels(got[0]), arrays[f"prepared_{key}"]), key


@pytest.mark.parametrize("case", CASES)
def test_the_runs_take_the_size_and_crop_of_their_references(check6, case):
    from gmlx.image.families.flux3_klein import edit

    arrays, m = check6
    row = m["results"]["runs"][case]
    refs = edit.prepare(_sources(arrays, row))
    assert list(edit.default_size(refs)) == row["size"]
    assert (None if refs.crop is None else list(refs.crop)) == row["crop"]
    width, height = row["size"]
    assert arrays[f"noise_{case}"].shape == (1, 128, height // 16,
                                             width // 16)
    for i, ref in enumerate(refs):
        lat = arrays[f"ref_latents_{case}_{i}"]
        assert lat.shape == (1, 128, ref.height // 16, ref.width // 16)


@pytest.mark.parametrize("case", CASES)
def test_the_reference_ids_are_the_ones_the_dit_read(check6, case):
    from gmlx.image.families.flux3_klein import family as ff
    from gmlx.image.families.flux3_klein.model import References

    arrays, m = check6
    n = len(m["results"]["runs"][case]["sources"])
    lats = [mx.array(arrays[f"ref_latents_{case}_{i}"]) for i in range(n)]
    got = References.of(lats, ff.CONSTANTS.values["image_rope_scale"])
    assert np.array_equal(np.asarray(got.ids, np.float32),
                          arrays[f"ref_ids_{case}"])
    # The last id of each token is -1 less the index of its reference.
    last = arrays[f"ref_ids_{case}"][:, 3]
    assert sorted(set(last.tolist()), reverse=True) == [
        -1.0 - i for i in range(n)]


def _files(gguf_dir, m):
    want = m["files"]["dit"]
    for p in image_dits(gguf_dir, FAMILY):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    if fx.file_identity(p)["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle read")
    vaes = sorted(vae_files(FAMILY), key=lambda p: p.suffix != ".gguf")
    if not vaes:
        pytest.skip("needs the FLUX.2 VAE under GMLX_TEST_IMAGE_DIR")
    return p, str(vaes[0])


@pytest.fixture(scope="module")
def pipe(check6, gguf_dir):
    from gmlx.image import api

    dit, vae = _files(gguf_dir, check6[1])
    return api.load(dit, vae=vae, log=lambda *_: None)


def _encode_gate(pipe) -> tuple:
    """(metric, gate) of check 5 for an encode in the VAE's dtype at 512
    pixels, the size of check 5 nearest the references."""
    t = committed(CHECK5)["tolerances"]["sizes"]["512x512"]
    if pipe.vae.dtype == mx.float32:
        return "max abs", float(t["encode_fp32_max_abs"])
    return "rel L2", float(t["encode_bf16_rel_l2"])


# A Mac of 16 GB, as most users run (tests/host_box.py).
@pytest.mark.integration
@pytest.mark.host_gb(16)
@pytest.mark.parametrize("case", CASES)
def test_reference_latents_of_the_vae_against_the_oracle(check6, pipe, case):
    from gmlx.image import api
    from gmlx.image.families.flux3_klein import edit

    arrays, m = check6
    refs = edit.prepare(_sources(arrays, m["results"]["runs"][case]))
    metric, gate = _encode_gate(pipe)
    for i, ref in enumerate(refs):
        got = np.array(api._reference_latents(pipe, ref, None))[None]
        want = arrays[f"ref_latents_{case}_{i}"]
        err = (fx.max_abs(got, want) if metric == "max abs"
               else fx.rel_l2(got, want))
        print(f"flux3 edit {case} reference {i}: {metric} {err:.2e} of "
              f"gate {gate:.2e}")
        assert err <= gate


def _run(pipe, arrays, m, case: str, order=None) -> dict:
    from gmlx.image import api
    from gmlx.image.conditioning import Conditioning

    steps = m["inputs"]["steps"]
    row = m["results"]["runs"][case]
    lats = [arrays[f"ref_latents_{case}_{i}"][0]
            for i in range(len(row["sources"]))]
    if order is not None:
        lats = [lats[i] for i in order]
    cond = Conditioning(mx.array(arrays["features"]), f"check6 edit {case}",
                        ref_latents=tuple(mx.array(x) for x in lats))
    want = {k: arrays[f"latents_{case}_step{k}"]
            for k in m["inputs"]["kept_steps"]}
    want[steps] = arrays[f"latents_{case}"]
    got = {}

    def on_step(i, x):
        if i + 1 in want:
            got[i + 1] = fx.rel_l2(np.array(x), want[i + 1])

    width, height = row["size"]
    api.sample(pipe, cond, height=height, width=width,
               noise=mx.array(arrays[f"noise_{case}"]), on_step=on_step)
    return got


def _gates(m, case: str) -> dict:
    by = m["tolerances"][f"by_step_rel_l2_{case}"]
    steps = m["inputs"]["steps"]
    assert len(by) == steps
    return {k: by[k - 1] for k in m["inputs"]["kept_steps"] + [steps]}


@pytest.mark.integration
@pytest.mark.host_gb(16)
@pytest.mark.parametrize("case", CASES)
def test_check6_edit_latents_from_the_conditioning_of_the_oracle(
        check6, pipe, case):
    arrays, m = check6
    gates = _gates(m, case)
    got = _run(pipe, arrays, m, case)
    print(f"flux3 check6 edit {case}: "
          + ", ".join(f"step {k} {v:.2e} ({v / gates[k]:.2f} of its gate)"
                      for k, v in sorted(got.items())))
    assert sorted(got) == sorted(gates)
    over = {k: (v, gates[k]) for k, v in got.items() if v > gates[k]}
    assert not over, over


@pytest.mark.integration
@pytest.mark.host_gb(16)
def test_check6_edit_fails_with_the_references_in_the_other_order(check6,
                                                                  pipe):
    arrays, m = check6
    gates = _gates(m, "two")
    got = _run(pipe, arrays, m, "two", order=(1, 0))
    share = {k: v / gates[k] for k, v in got.items()}
    print(f"flux3 check6 edit two, other order: {share}")
    assert min(share.values()) > 1.0, share

