"""Patch table entries from ComfyUI patches: the weight key, the offset
band and the adapter tuple give the table a LoRA file gives. CPU."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from gmlx.image import api, lora
from gmlx.image.families.z_image.lora_map import lora_map
from gmlx.image.registry import get_family
from image_fixtures import TINY_Z_CONFIG, tiny_z_model
from lora_fixtures import make_pair, tiny_z_adapter, write_adapter


def _pipe() -> api.ImagePipeline:
    m, _ = tiny_z_model()
    return api.ImagePipeline(dit=m, family=get_family("z_image"),
                             variant="turbo")


def _patches(pipe, stack, prefix="diffusion_model.") -> list:
    """The patch list ComfyUI would hold for (adapter, strength) rows: one
    row per weight key and offset, with the pair registered from the
    adapter tuple (up, down, alpha, mid, dora_scale, reshape)."""
    km = lora_map(TINY_Z_CONFIG)
    rows: dict = {}
    for adapter, strength in stack:
        for target, (down, up, alpha) in adapter.items():
            module, band = km.targets[target]
            key = lora.register_patch(pipe, up, down, alpha, None, None, None)
            rows.setdefault((module, band), []).append(
                {"adapter": key, "strength": strength, "strength_model": 1.0})
    return [{"key": f"{prefix}{module}.weight",
             "band": None if band is None else list(band), "patches": p}
            for (module, band), p in rows.items()]


def test_a_patch_list_gives_the_table_of_the_same_files(tmp_path):
    one, two = tiny_z_adapter(1, 4.0, wide=True), tiny_z_adapter(2, 4.0)
    files = _pipe()
    a1 = lora.load_adapter(files, write_adapter(
        tmp_path / "one.safetensors", "kohya", one))
    a2 = lora.load_adapter(files, write_adapter(
        tmp_path / "two.safetensors", "onetrainer", two))
    for stack, rows in (([(one, 0.8)], [(a1, 0.8)]),
                        ([(one, 0.8), (two, -0.3)], [(a1, 0.8), (a2, -0.3)])):
        want = api.define_patch_table(files, lora.table_entries(rows))
        for prefix in ("diffusion_model.", "model.diffusion_model.", ""):
            pipe = _pipe()
            entries = lora.plans_from_patches(
                pipe.family, _patches(pipe, stack, prefix))
            assert api.define_patch_table(pipe, entries) == want
            # The pairs are the ones the files hold, by content.
            assert set(pipe.adapters._items) <= set(files.adapters._items)
    entries = lora.plans_from_patches(files.family,
                                      _patches(files, [(one, 0.8)]))
    qkv = [e for e in entries if e[0] == "layers.1.attention.qkv"]
    assert sorted(e[1] for e in qkv) == [(0, 0, 64), (0, 64, 64),
                                         (0, 128, 64)]
    assert all(len(e[2]) == 1 and e[2][0][1] == 0.8 for e in qkv)


def test_an_offset_key_and_tensor_forms_are_read(tmp_path):
    pipe = _pipe()
    rng = np.random.default_rng(0)
    d, u, _ = make_pair(rng, 64, 64, 2)
    h = lora.register_patch(pipe, u, d, 3.0)
    assert h == lora.pair_hash(mx.array(d), mx.array(u), 3.0)
    # Alpha as a tensor, and the factors of a 1x1 kernel.
    assert lora.register_patch(pipe, u.reshape(64, 2, 1, 1),
                               d.reshape(2, 64, 1, 1),
                               np.float32(3.0)) == h
    assert lora.register_patch(pipe, mx.array(u), mx.array(d),
                               mx.array([3.0])) == h
    assert lora.register_patch(pipe, u, d) != h and len(pipe.adapters) == 2
    entries = lora.plans_from_patches(pipe.family, [
        {"key": "diffusion_model.layers.0.attention.qkv.weight",
         "offset": (0, 128, 64),
         "patches": [{"adapter": h, "strength": 1}]}])
    assert entries == [("layers.0.attention.qkv", (0, 128, 64), [(h, 1.0)])]
    assert api.define_patch_table(pipe, entries) != api.EMPTY_TABLE


def test_patches_gmlx_does_not_apply_are_refused_by_name():
    pipe = _pipe()
    rng = np.random.default_rng(0)
    d, u, _ = make_pair(rng, 64, 64, 2)
    extra = mx.ones((2, 2))
    for kw, kind in ((dict(mid=extra), "Tucker LoCon"),
                     (dict(dora_scale=extra), "DoRA"),
                     (dict(reshape=(64, 64)), "reshape")):
        with pytest.raises(lora.UnsupportedAdapterError, match=kind) as e:
            lora.register_patch(pipe, u, d, 1.0, **kw)
        assert e.value.kind == kind
    assert len(pipe.adapters) == 0
    h = lora.register_patch(pipe, u, d, 1.0)
    row = {"key": "diffusion_model.layers.0.attention.out.weight",
           "band": None, "patches": [{"adapter": h, "strength": 1.0}]}
    scaled = dict(row, patches=[{"adapter": h, "strength": 1.0,
                                 "strength_model": 0.5}])
    with pytest.raises(lora.UnsupportedAdapterError,
                       match="scales the base weight by 0.5"):
        lora.plans_from_patches(pipe.family, [scaled])
    with pytest.raises(lora.UnsupportedAdapterError, match="bias patch"):
        lora.plans_from_patches(pipe.family, [dict(
            row, key="diffusion_model.final_layer.linear.bias")])
    # A kind other than plain lora, and a field this code does not read,
    # are refused by name and never applied as plain LoRA.
    one = {"adapter": h, "strength": 1.0}
    for kind in ("diff", "set", "model_as_lora", "loha", "lokr", "glora",
                 "oft", "boft"):
        with pytest.raises(lora.UnsupportedAdapterError,
                           match=f"is a {kind} patch") as e:
            lora.plans_from_patches(pipe.family, [dict(
                row, patches=[dict(one, kind=kind)])])
        assert e.value.kind == kind
    assert lora.plans_from_patches(pipe.family, [dict(
        row, patches=[dict(one, kind="lora")])]) == \
        lora.plans_from_patches(pipe.family, [row])
    for bad, names in ((dict(row, function=len), "field function"),
                       (dict(row, hook=1, pad=2), "fields hook, pad"),
                       (dict(row, patches=[dict(one, dora_scale=extra)]),
                        "field dora_scale"),
                       (dict(row, patches=[dict(one, mid=extra, mode=1)]),
                        "fields mid, mode")):
        with pytest.raises(lora.UnsupportedAdapterError, match=names) as e:
            lora.plans_from_patches(pipe.family, [bad])
        assert e.value.kind == "patch field" \
            and "layers.0.attention.out" in str(e.value)
    with pytest.raises(lora.UnsupportedAdapterError, match="which differ"):
        lora.plans_from_patches(pipe.family, [dict(
            row, band=(0, 0, 64), offset=(0, 64, 64))])
    assert lora.plans_from_patches(pipe.family, [dict(
        row, band=[0, 0, 64], offset=(0, 0, 64))])[0][1] == (0, 0, 64)
    other = lora.plans_from_patches(pipe.family, [dict(
        row, key="diffusion_model.double_blocks.0.img_attn.qkv.weight")])
    with pytest.raises(ValueError, match="not a module of this DiT"):
        api.define_patch_table(pipe, other)
