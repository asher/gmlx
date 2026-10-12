"""Caches that outlive a model instance (the APC disk tier, a distill cache)
key the rope scaling a load resolves: the GGUF header and the options,
plus the rope revision. Tiny GGUF headers, no weights."""

from __future__ import annotations

import numpy as np

from gmlx.load import rope_scaling as rs
from gmlx.load.rope_scaling import RopeOverride


def _gguf(path, **rope):
    from gguf import GGUFWriter

    w = GGUFWriter(str(path), "qwen3")
    w.add_uint32("qwen3.context_length", 32768)
    w.add_float32("qwen3.rope.freq_base", 1000000.0)
    for k, v in rope.items():
        key = f"qwen3.rope.scaling.{k}"
        if isinstance(v, str):
            w.add_string(key, v)
        else:
            w.add_float32(key, v)
    w.add_tensor("token_embd.weight", np.zeros((4, 8), dtype=np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def test_the_key_follows_the_resolved_scaling(tmp_path, monkeypatch):
    plain = _gguf(tmp_path / "plain.gguf")
    yarn = _gguf(tmp_path / "yarn.gguf", type="yarn", factor=4.0)
    # A plain rope keeps the name as it was.
    assert rs.cache_suffix(plain, None) == ""
    assert rs.cache_suffix(plain, RopeOverride()) == ""
    # Scaling from the header alone keys, as the header is applied now.
    header = rs.cache_suffix(yarn, None)
    assert header.startswith("#rope-")
    # The same scaling from options keys the same; another factor does not.
    assert rs.cache_suffix(plain, RopeOverride(
        rope_scaling="yarn", rope_scale=4.0)) == header
    assert rs.cache_suffix(yarn, RopeOverride(rope_scale=2.0)) != header
    # A change in how a scaling becomes a table (a header fix) moves it.
    monkeypatch.setattr(rs, "ROPE_REVISION", rs.ROPE_REVISION + 1)
    assert rs.cache_suffix(yarn, None) != header
    # An unreadable path resolves from the options alone.
    assert rs.cache_suffix(tmp_path / "absent.gguf", None) == ""


def test_distill_run_fingerprint_keys_the_teacher_rope(tmp_path):
    from gmlx.distill import teacher as _teacher

    def fp(path):
        opts = _teacher.CacheOptions(teacher=path, corpus=str(tmp_path / "c"),
                                     out=str(tmp_path / "o"))
        return _teacher.run_fingerprint(opts, "x", 1, 1, None)["rope"]

    assert fp(_gguf(tmp_path / "plain.gguf")) is None
    a = fp(_gguf(tmp_path / "a.gguf", type="yarn", factor=4.0))
    b = fp(_gguf(tmp_path / "b.gguf", type="yarn", factor=2.0))
    assert a and b and a != b
