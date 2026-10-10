"""The image activation policy, header classification, and the image
hints that text-load refusals append. CPU."""
from __future__ import annotations

import numpy as np
import pytest
from image_fixtures import write_gguf

import mlx.core as mx

from gmlx.image import dtypes, preflight
from gmlx.image.dtypes import ImageDtypeError, image_activation


def test_native_bf16_chips_run_bf16_everywhere():
    for gen in (0, 15, 16, 17):
        act = image_activation("z_image", gen=gen)
        assert (act.dit, act.vae) == (mx.bfloat16, mx.bfloat16)
        act = image_activation("qwen_image21", gen=gen)
        assert act.dit == mx.bfloat16
        assert dtypes.vae_dtype(gen=gen) == mx.bfloat16


def test_m1_and_m2_need_the_override(monkeypatch):
    monkeypatch.delenv(dtypes.ALLOW_FP16_ENV, raising=False)
    for gen, chip in ((13, "M1"), (14, "M2")):
        with pytest.raises(ImageDtypeError) as e:
            image_activation("z_image", gen=gen)
        msg = str(e.value)
        assert msg == (f"Z-Image runs in float16 on the {chip} chip, which "
                       "misses the reference checks, so its images differ "
                       "from an M3 or newer. Set GMLX_IMAGE_ALLOW_FP16=1 to "
                       "run it anyway")
        act = image_activation("z_image", gen=gen, allow=True)
        assert (act.dit, act.vae) == (mx.float16, mx.float32)
        assert dtypes.vae_dtype(gen=gen) == mx.float32
    monkeypatch.setenv(dtypes.ALLOW_FP16_ENV, "1")
    assert image_activation("z_image", gen=13).dit == mx.float16
    monkeypatch.setenv(dtypes.ALLOW_FP16_ENV, "0")
    with pytest.raises(ImageDtypeError):
        image_activation("z_image", gen=13)


def test_families_without_fp16_are_refused_even_with_the_override(
        monkeypatch):
    monkeypatch.setenv(dtypes.ALLOW_FP16_ENV, "1")
    with pytest.raises(ImageDtypeError,
                       match="Qwen-Image 2.1 needs bfloat16.*never runs in "
                             "float16"):
        image_activation("qwen_image21", gen=14)


def test_the_vae_reader_default_follows_the_policy(monkeypatch):
    from gmlx.image.vae import reader
    from gmlx.load import dtypes as load_dtypes

    monkeypatch.setattr(load_dtypes, "gpu_arch_gen", lambda: 13)
    assert reader.resolve_dtype(None) == mx.float32
    monkeypatch.setattr(load_dtypes, "gpu_arch_gen", lambda: 17)
    assert reader.resolve_dtype(None) == mx.bfloat16


# Classification


def _gguf(path, *, arch, kind=None, tensors=None):
    import gguf

    w = gguf.GGUFWriter(str(path), arch or "none")
    if arch is None:
        w.kv_data[0].pop("general.architecture", None)
    if kind:
        w.add_type(kind)
    for name, arr in (tensors or {"w": np.zeros((4, 4), np.float32)}).items():
        w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def _relative(path) -> str:
    """A no-arch GGUF with the names of a Z-Image relative, so the Z-Image
    fingerprint refuses it by name."""
    z = np.zeros((2, 2), np.float32)
    return write_gguf(path, {
        "layers.0.attention.qkv.weight": (z, "F32"),
        "cap_embedder.1.weight": (z, "F32"),
        "siglip_embedder.0.weight": (z, "F32")}, arch=None)


def _vae_names(path) -> str:
    """A no-arch GGUF with FLUX.1 autoencoder names and tiny shapes."""
    from mlx.utils import tree_flatten

    from gmlx.image.vae import flux1_ae

    cfg = flux1_ae.Flux1AEConfig(3, 3, 16, (32, 64), 1, 32, 1e-6)
    tensors = {}
    for name, p in tree_flatten(flux1_ae.Flux1AE(cfg).parameters()):
        shape = p.shape
        if len(shape) == 4:
            shape = (shape[0], shape[3], shape[1], shape[2])
        tensors[name] = np.zeros(shape, np.float32)
    return _gguf(path, arch=None, tensors=tensors)


def test_classify_reads_type_arch_and_names(tmp_path):
    cases = {
        "text": _gguf(tmp_path / "t.gguf", arch="llama"),
        "image_dit": _gguf(tmp_path / "d.gguf", arch="lumina2"),
        "vae": _gguf(tmp_path / "v.gguf", arch="autoencoder", kind="vae"),
        "mmproj": _gguf(tmp_path / "m.gguf", arch="clip", kind="mmproj"),
        "adapter": _gguf(tmp_path / "a.gguf", arch="llama", kind="adapter"),
        "unknown": _gguf(tmp_path / "u.gguf", arch="nosuch"),
    }
    for kind, path in cases.items():
        assert preflight.classify(path) == kind, kind
    assert preflight.classify(_relative(tmp_path / "r.gguf")) == "image_dit"
    assert preflight.classify(_vae_names(tmp_path / "n.gguf")) == "vae"
    assert preflight.classify(
        _gguf(tmp_path / "q.gguf", arch="qwen_image")) == "image_dit"


def test_arch_gate_appends_the_image_hint():
    from gmlx.load.arch_table import UnsupportedArchError, gate

    with pytest.raises(UnsupportedArchError) as e:
        gate("lumina2")
    assert str(e.value).endswith(preflight.IMAGE_HINT)
    with pytest.raises(UnsupportedArchError) as e:
        gate("autoencoder")
    assert str(e.value).endswith(preflight.VAE_HINT)
    with pytest.raises(UnsupportedArchError) as e:
        gate("nosuch")
    assert str(e.value).endswith("See docs/arch-coverage.md.")


def test_missing_arch_preflight_appends_the_image_hint(tmp_path):
    from gmlx.load.preflight import preflight as text_preflight

    with pytest.raises(ValueError) as e:
        text_preflight(_relative(tmp_path / "r.gguf"))
    assert str(e.value).endswith(preflight.IMAGE_HINT)
    with pytest.raises(ValueError) as e:
        text_preflight(_vae_names(tmp_path / "v.gguf"))
    assert str(e.value).endswith(preflight.VAE_HINT)
    with pytest.raises(ValueError) as e:
        text_preflight(_gguf(tmp_path / "x.gguf", arch=None))
    assert str(e.value).endswith("can't detect arch")


def test_hints_never_raise(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("broken header")

    monkeypatch.setattr(preflight, "classify", boom)
    assert preflight.file_hint("/nonexistent.gguf") == ""
    from gmlx.image import registry

    monkeypatch.setattr(registry, "is_image_arch", boom)
    assert preflight.arch_hint("lumina2") == ""
    assert preflight.arch_hint("autoencoder") == preflight.VAE_HINT
