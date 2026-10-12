"""The Qwen-Image 2.1 VAE on the official weights: the file has the
committed header inventory and sha256, a GGUF packed from it keeps F32 and
loads to the same weights bit for bit, a packed GGUF found beside it does
too, and the header classifies as a VAE. Only weights are loaded, twice in
float32, which is about 2.5 GiB, and nothing is decoded or encoded.
Skipped unless GMLX_TEST_IMAGE_DIR holds the Qwen-Image 2.1 VAE
safetensors."""
from __future__ import annotations

import os

import pytest
from image_dir import sha256, vae_files
from image_fixtures import load_json

import mlx.core as mx
from mlx.utils import tree_flatten

pytestmark = pytest.mark.integration

FAMILY = "qwen_image21"
HEADER = "qwen_image21_vae.header.json"


@pytest.fixture(scope="module")
def sources():
    files = vae_files(FAMILY)
    st = [p for p in files if p.suffix == ".safetensors"]
    if not st:
        pytest.skip("no Qwen-Image 2.1 VAE safetensors under "
                    "GMLX_TEST_IMAGE_DIR")
    return st[0], [p for p in files if p.suffix == ".gguf"]


def _weights(path) -> dict:
    from gmlx.image.vae import load_vae

    vae = load_vae(path, family=FAMILY, dtype="fp32")
    return dict(tree_flatten(vae.parameters()))


def _same(a: dict, b: dict) -> list:
    assert a.keys() == b.keys()
    return [k for k in a if not (a[k].dtype == b[k].dtype
                                 and mx.array_equal(a[k], b[k]).item())]


def test_the_file_is_the_inventoried_one(sources):
    from gmlx.image.vae import read_header

    st, _ = sources
    want = load_json("manifest.json")[HEADER]
    head = read_header(str(st))
    got = {n: (list(head.shapes[n]), head.dtypes[n]) for n in head.shapes}
    assert got == {n: (s, d) for n, s, d in load_json(HEADER)["tensors"]}
    assert os.path.getsize(st) == want["size"]
    assert sha256(st) == want["sha256"]


def test_a_packed_gguf_keeps_f32_and_loads_the_safetensors_weights(
        sources, tmp_path):
    from gmlx.image import preflight
    from gmlx.image.vae import pack_vae, read_header

    st, _ = sources
    cfg = st.parent / "config.json"
    dst = tmp_path / "vae.gguf"
    rep = pack_vae(str(st), str(dst), source_repo="test",
                   source_revision="test",
                   config_json=cfg.read_text() if cfg.is_file() else "{}")
    assert rep.lossless and rep.family == FAMILY and rep.dtype == "source"
    assert rep.tensors == 238
    assert rep.payload_bytes == load_json(HEADER)["payload_bytes"]
    assert set(read_header(str(dst)).dtypes.values()) == {"F32"}
    assert preflight.classify(str(dst)) == "vae"
    assert preflight.file_hint(str(dst)) == preflight.VAE_HINT
    a = _weights(st)
    assert len(a) == 226 and not any(".time_conv." in k for k in a)
    assert _same(a, _weights(dst)) == []


def test_a_packed_gguf_beside_the_source_matches_it(sources):
    from gmlx.image.vae import read_header

    st, ggufs = sources
    digest = sha256(st)
    packed = [g for g in ggufs if read_header(str(g)).metadata.get(
        "gmlx.vae.source_sha256") == digest]
    if not packed:
        pytest.skip("no GGUF packed from this VAE under GMLX_TEST_IMAGE_DIR")
    ref = _weights(st)
    for g in packed:
        assert _same(ref, _weights(g)) == [], g.name
