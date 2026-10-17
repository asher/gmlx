"""Check 6 of the image acceptance gate for FLUX.2 klein 9B KV: whole
sampling runs on injected noise against BFL's own, for text to image and
for edits with one and with four references. An edit is the reference's
cached pass, denoise_cached: its first step reads the references, and
every later step reads their cached K and V.

gmlx samples through the KV row from the oracle's noise and conditioning,
its features and, for an edit, its float32 encode of each reference, with
KV's defaults, on the reference's schedule. The latents after each kept
step and after the last stay within twice the oracle's own bf16 against
fp32 gap after that step.

Skipped unless the manifests of scripts/image_oracle/check6_flux2_klein.py
and check6_flux2_klein_edit.py, each run with --kv, are committed,
GMLX_IMAGE_FIXTURES holds their arrays and KQUANT_TEST_GGUF_DIR the KV
GGUF the oracle read. The runs declare a Mac of 24 GB."""
from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from image_fixtures import image_dits
from oracle_fixture import ORACLE, committed, fixture, fx

FAMILY = "flux2_klein_9b_kv"
CASES = {"text to image": "check6_flux2_klein_9b_kv_512",
         "one reference": "check6_flux2_klein_9b_kv_edit_r1",
         "four references": "check6_flux2_klein_9b_kv_edit_r4"}
REFS = {"text to image": 0, "one reference": 1, "four references": 4}
DIT = "Flux-2-Klein-9B-KV-Q8_0.gguf"
REFERENCE_COMMIT = "50fe5162777813d869182b139e83b10743caef15"


def _manifest(case: str) -> dict:
    name = CASES[case]
    if not (ORACLE / f"{name}.manifest.json").is_file():
        pytest.skip(f"{name}.manifest.json is not committed: run the check "
                    "6 oracles of FLUX.2 klein with --kv on the KV GGUF and "
                    "commit the manifest")
    return committed(name)


@pytest.fixture(scope="module")
def pipe(gguf_dir):
    from gmlx.image import api

    want = None
    for case in CASES:
        if (ORACLE / f"{CASES[case]}.manifest.json").is_file():
            want = committed(CASES[case])["files"]["dit"]
            break
    if want is None:
        pytest.skip("no check 6 manifest of FLUX.2 klein 9B KV is committed")
    for p in image_dits(gguf_dir, FAMILY):
        if Path(p).name == want["name"] and os.path.getsize(p) == want["size"]:
            break
    else:
        pytest.skip(f"{want['name']} is not under KQUANT_TEST_GGUF_DIR")
    if fx.file_identity(p)["sha256"] != want["sha256"]:
        pytest.fail(f"{p} is not the file the oracle read")
    return api.load(p, log=lambda *_: None)


@pytest.mark.integration
@pytest.mark.host_gb(24)
@pytest.mark.parametrize("case", list(CASES))
def test_check6_kv_latents(pipe, case):
    from gmlx.image import api
    from gmlx.image.conditioning import Conditioning

    m = _manifest(case)
    arrays, _ = fixture(m["name"])
    ins, run = m["inputs"], m["results"]["run"]
    assert m["family"] == FAMILY and m["files"]["dit"]["name"] == DIT
    assert ins["reference_commit"] == REFERENCE_COMMIT
    assert ins["model_info"] == "flux.2-klein-9b-kv"
    assert (pipe.family.name, pipe.variant) == (FAMILY, ins["variant"])
    by_step = m["tolerances"]["by_step_rel_l2"]
    assert by_step == [2 * g for g in run["bf16_vs_fp32_by_step"]]
    steps = ins["steps"]
    gates = {k: by_step[k - 1] for k in ins["kept_steps"] + [steps]}
    refs = tuple(mx.array(arrays[f"ref_latents_{i}"][0])
                 for i in range(REFS[case]))
    assert f"ref_latents_{len(refs)}" not in arrays
    cond = Conditioning(mx.array(arrays["features"]), f"check6 kv {case}",
                        ref_latents=refs)
    width, height = run.get("size", (ins.get("size"),) * 2)
    sigmas = pipe.sigmas(steps, (height // 16) * (width // 16))
    assert np.array_equal(sigmas, arrays["schedule"])
    want = {k: arrays[f"latents_step{k}"] for k in ins["kept_steps"]}
    want[steps] = arrays["latents"]
    got = {}

    def on_step(i, x):
        if i + 1 in want:
            got[i + 1] = fx.rel_l2(np.array(x), want[i + 1])

    api.sample(pipe, cond, height=height, width=width, sigmas=sigmas,
               noise=mx.array(arrays["noise"]), on_step=on_step)
    print(f"flux2 check6 kv {case}: " + ", ".join(
        f"step {k} {v:.2e} ({v / gates[k]:.2f} of its gate)"
        for k, v in sorted(got.items())))
    assert sorted(got) == sorted(gates)
    over = {k: (v, gates[k]) for k, v in got.items() if v > gates[k]}
    assert not over, over
