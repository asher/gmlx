"""The FLUX.1 VAE on the official Z-Image weights: the safetensors file and
a GGUF packed from it load to the same weights bit for bit, a packed GGUF
found beside it does too, and the header classifies as a VAE. Skipped
unless GMLX_TEST_IMAGE_DIR holds the Z-Image VAE safetensors."""
from __future__ import annotations

import pytest
from image_dir import sha256, vae_files

import mlx.core as mx
from mlx.utils import tree_flatten

pytestmark = pytest.mark.integration

FAMILY = "z_image"


@pytest.fixture(scope="module")
def sources():
    files = vae_files(FAMILY)
    st = [p for p in files if p.suffix == ".safetensors"]
    if not st:
        pytest.skip("no Z-Image VAE safetensors under GMLX_TEST_IMAGE_DIR")
    return st[0], [p for p in files if p.suffix == ".gguf"]


def _weights(path) -> dict:
    from gmlx.image.vae import load_vae

    vae = load_vae(path, family=FAMILY, dtype="bf16")
    return dict(tree_flatten(vae.parameters()))


def _same(a: dict, b: dict) -> list:
    assert a.keys() == b.keys()
    return [k for k in a if not (a[k].dtype == b[k].dtype
                                 and mx.array_equal(a[k], b[k]).item())]


def test_a_packed_gguf_loads_the_safetensors_weights(sources, tmp_path):
    from gmlx.image import preflight
    from gmlx.image.vae import pack_vae

    st, _ = sources
    cfg = st.parent / "config.json"
    dst = tmp_path / "vae.gguf"
    rep = pack_vae(str(st), str(dst), source_repo="test",
                   source_revision="test",
                   config_json=cfg.read_text() if cfg.is_file() else "{}")
    assert rep.lossless and rep.family == FAMILY
    assert rep.tensors == 244
    assert preflight.classify(str(dst)) == "vae"
    assert preflight.file_hint(str(dst)) == preflight.VAE_HINT
    assert _same(_weights(st), _weights(dst)) == []


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
