"""Check 6 of the image acceptance gate for FLUX.2 klein edits: a whole
sampling run with one reference image, on injected noise, against BFL's
own encode_image_refs and denoise.

The preparation of a reference, its size check, pixel cap and grid crop,
equals the reference's for every source the oracle prepared, bit for
bit, and refuses what the reference refuses. The reference tokens the
DiT reads, their bf16 rounding and their ids, are those the reference's
DiT read. The reference latents of gmlx's VAE are held to the oracle's
float32 encode by the gate check 5 sets for an encode of that dtype at
512 pixels. From the oracle's conditioning, its features and reference
latents, the latents after each kept step and after the last stay within
twice the oracle's own bf16 against fp32 gap after that step. The run
without its reference misses those gates.

Skipped unless the manifest of scripts/image_oracle/check6_flux2_klein_edit.py
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

NAME = "check6_flux2_klein_edit"
FAMILY = "flux2_klein"
CHECK5 = "check5_flux3_klein_vae"


@pytest.fixture(scope="module")
def check6():
    if not (ORACLE / f"{NAME}.manifest.json").is_file():
        pytest.skip(f"{NAME}.manifest.json is not committed: run "
                    "scripts/image_oracle/check6_flux2_klein_edit.py on the "
                    "FLUX.2 klein GGUF and commit the manifest")
    arrays, _ = fixture(NAME)
    return arrays, committed(NAME)


@pytest.mark.parametrize("rule", ["one", "several"])
def test_the_references_are_prepared_as_the_reference_prepares_them(
        check6, rule):
    from gmlx.image.families.flux2_klein import edit

    arrays, m = check6
    rows = {k: r for k, r in m["inputs"]["prepared"].items()
            if k.split("_", 1)[0] == rule}
    assert len(rows) >= 4
    count = 1 if rule == "one" else 2
    for key, row in rows.items():
        src = arrays[f"source_{key.split('_', 1)[1]}"]
        if "error" in row:
            with pytest.raises(edit.BadReference):
                edit.prepare([src] * count)
            continue
        got = edit.prepare([src] * count)
        assert got.crop is None
        assert edit.default_size(got) == tuple(row["size"]), key
        for ref in got:
            assert (ref.pixels[..., 3] == 255).all()
            assert np.array_equal(ref.pixels[..., :3],
                                  arrays[f"prepared_{key}"]), key
    assert any("error" in r for r in rows.values()) == (rule == "one")


def test_the_run_takes_the_size_of_its_reference(check6):
    from gmlx.image.families.flux2_klein import edit

    arrays, m = check6
    row = m["results"]["run"]
    refs = edit.prepare([arrays[f"source_{k}"] for k in row["sources"]])
    assert list(edit.default_size(refs)) == row["size"]
    width, height = row["size"]
    assert arrays["noise"].shape == (1, 128, height // 16, width // 16)
    assert arrays["ref_latents_0"].shape == (1, 128, refs[0].height // 16,
                                             refs[0].width // 16)


def test_the_reference_rows_are_the_ones_the_dit_read(check6):
    """The tokens of the oracle's latents in bf16, with the ids of index
    0, are the reference rows its DiT read."""
    from gmlx.image.families.flux2_klein import model as fm

    arrays, _ = check6
    got = fm.Flux2KleinDiT.references(None, [mx.array(arrays["ref_latents_0"])])
    assert np.array_equal(np.array(got.tokens[0].astype(mx.float32)),
                          arrays["ref_tokens"])
    assert np.array_equal(np.asarray(got.ids, np.float32), arrays["ref_ids"])
    assert set(arrays["ref_ids"][:, 0].tolist()) == {10.0}


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
    pixels, the size of check 5 nearest the reference. The VAE is FLUX.3
    klein's, so its check 5 is this one's."""
    t = committed(CHECK5)["tolerances"]["sizes"]["512x512"]
    if pipe.vae.dtype == mx.float32:
        return "max abs", float(t["encode_fp32_max_abs"])
    return "rel L2", float(t["encode_bf16_rel_l2"])


@pytest.mark.integration
def test_reference_latents_of_the_vae_against_the_oracle(check6, pipe):
    from gmlx.image import api
    from gmlx.image.families.flux2_klein import edit

    arrays, m = check6
    (ref,) = edit.prepare([arrays[f"source_{k}"]
                           for k in m["results"]["run"]["sources"]])
    metric, gate = _encode_gate(pipe)
    got = np.array(api._reference_latents(pipe, ref, None))[None]
    want = arrays["ref_latents_0"]
    err = (fx.max_abs(got, want) if metric == "max abs"
           else fx.rel_l2(got, want))
    print(f"flux2 edit reference: {metric} {err:.2e} of gate {gate:.2e}")
    assert err <= gate


def _run(pipe, arrays, m, refs: bool = True) -> dict:
    from gmlx.image import api
    from gmlx.image.conditioning import Conditioning

    steps = m["inputs"]["steps"]
    row = m["results"]["run"]
    lats = (mx.array(arrays["ref_latents_0"][0]),) if refs else ()
    cond = Conditioning(mx.array(arrays["features"]), "check6 flux2 edit",
                        ref_latents=lats)
    want = {k: arrays[f"latents_step{k}"] for k in m["inputs"]["kept_steps"]}
    want[steps] = arrays["latents"]
    got = {}

    def on_step(i, x):
        if i + 1 in want:
            got[i + 1] = fx.rel_l2(np.array(x), want[i + 1])

    width, height = row["size"]
    sigmas = pipe.sigmas(steps, (height // 16) * (width // 16))
    api.sample(pipe, cond, height=height, width=width, sigmas=sigmas,
               noise=mx.array(arrays["noise"]), on_step=on_step)
    return got


def _gates(m) -> dict:
    by = m["tolerances"]["by_step_rel_l2"]
    steps = m["inputs"]["steps"]
    assert len(by) == steps
    return {k: by[k - 1] for k in m["inputs"]["kept_steps"] + [steps]}


@pytest.mark.integration
def test_check6_edit_latents_from_the_conditioning_of_the_oracle(check6,
                                                                 pipe):
    arrays, m = check6
    gates = _gates(m)
    got = _run(pipe, arrays, m)
    print("flux2 check6 edit: "
          + ", ".join(f"step {k} {v:.2e} ({v / gates[k]:.2f} of its gate)"
                      for k, v in sorted(got.items())))
    assert sorted(got) == sorted(gates)
    over = {k: (v, gates[k]) for k, v in got.items() if v > gates[k]}
    assert not over, over


@pytest.mark.integration
def test_check6_edit_fails_without_its_reference(check6, pipe):
    arrays, m = check6
    gates = _gates(m)
    got = _run(pipe, arrays, m, refs=False)
    share = {k: v / gates[k] for k, v in got.items()}
    print(f"flux2 check6 edit without its reference: {share}")
    assert min(share.values()) > 1.0, share
