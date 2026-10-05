"""The text encoder halves of an image adapter: applied through the text
adapter wrap on an owned encoder, refused by name on a borrowed one. CPU."""
from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from gmlx.image import api, lora
from gmlx.image import text_encoder as te
from gmlx.image.registry import get_family
from image_fixtures import tiny_z_model
from lora_fixtures import (dialect_keys, make_pair, rel_l2, tiny_z_adapter,
                           write_keys)
from test_image_text_encoder import _CharTokenizer, _spec, _tiny_qwen3

TEXT_TOL = 2e-5
Q = "model.layers.0.self_attn.q_proj"          # 64 -> 64
K = "model.layers.2.self_attn.k_proj"          # 64 -> 32
UP = "model.layers.3.mlp.up_proj"              # 64 -> 96


@pytest.fixture(autouse=True)
def _cpu_stream():
    # The CPU stream keeps fp32 matmuls exact. On the GPU they run at TF32,
    # whose 1e-3 error is above the gap to the merged weights.
    with mx.stream(mx.cpu):
        yield


def _text_pairs(seed: int, alpha=None) -> dict:
    rng = np.random.default_rng(seed)
    return {Q: make_pair(rng, 64, 64, 4, alpha),
            K: make_pair(rng, 32, 64, 2, alpha),
            UP: make_pair(rng, 96, 64, 2, alpha)}


def _write(path, text: dict, dit: dict | None = None) -> str:
    """A kohya file with ComfyUI's spelling of the text encoder keys."""
    keys = dialect_keys("kohya", dit or {})
    for module, (down, up, alpha) in text.items():
        stem = f"text_encoders.qwen3_4b.transformer.{module}"
        keys[stem + ".lora_down.weight"] = down
        keys[stem + ".lora_up.weight"] = up
        if alpha is not None:
            keys[stem + ".alpha"] = np.asarray(alpha, np.float32)
    return write_keys(path, keys)


def _pipe(owned: bool = True) -> api.ImagePipeline:
    m, cfg = _tiny_qwen3()
    spec = _spec()
    enc = (te.EncoderHandle(m, _CharTokenizer(), spec, owned=True,
                            source_key="tiny")
           if owned else te.borrow_encoder(m, _CharTokenizer(), cfg, spec))
    dit, _ = tiny_z_model()
    return api.ImagePipeline(dit=dit, family=get_family("z_image"),
                             variant="turbo", encoder=enc)


def _merged_hidden(stack, prompt: str = "ab") -> np.ndarray:
    """The tap of the tiny encoder with (pairs, strength) rows merged into
    its weights in float64."""
    from mlx.utils import tree_flatten

    m, _ = _tiny_qwen3()
    params = dict(tree_flatten(m.parameters()))
    upd = {}
    for pairs, strength in stack:
        for module, (down, up, alpha) in pairs.items():
            name = f"{module}.weight"
            w = upd.get(name, np.array(params[name], np.float64))
            f = strength * (1.0 if alpha is None else alpha / down.shape[0])
            upd[name] = w + f * (up.astype(np.float64)
                                 @ down.astype(np.float64))
    m.load_weights([(k, mx.array(v.astype(np.float32)))
                    for k, v in upd.items()], strict=False)
    enc = te.EncoderHandle(m, _CharTokenizer(), _spec(), owned=True)
    return np.array(enc.encode(prompt).hidden[0], np.float64)


def _wrapped(model) -> list:
    from gmlx.load.modules import LoRAKQuantLinear

    return sorted(p for p, m in model.named_modules()
                  if isinstance(m, LoRAKQuantLinear))


def test_text_halves_apply_on_an_owned_encoder(tmp_path):
    one, two = _text_pairs(1, 6.0), _text_pairs(2)
    pipe = _pipe()
    enc = pipe.encoder
    a1 = lora.load_adapter(pipe, _write(tmp_path / "one.safetensors", one,
                                        tiny_z_adapter(1, 4.0)))
    a2 = lora.load_adapter(pipe, _write(tmp_path / "two.safetensors", two))
    assert len(a1.dit) == 8 and sorted(p for p, _ in a1.text) == [Q, K, UP]
    assert a2.dit == () and lora.table_entries([(a2, 1.0)]) == []
    base = np.array(enc.encode("ab").hidden[0], np.float64)
    before = api.encode_text(pipe, "ab")
    assert enc.adapters == "" and _wrapped(enc.model) == []

    rows = [(a1, 0.8), (a2, -0.5)]
    assert lora.apply_text_adapters(pipe, rows) == 2
    assert _wrapped(enc.model) == [Q, K, UP]
    got = np.array(enc.encode("ab").hidden[0], np.float64)
    want = _merged_hidden([(one, 0.8), (two, -0.5)])
    assert rel_l2(got, want) <= TEXT_TOL
    assert rel_l2(base, want) > 1e-2
    # The conditioning key carries the adapter state.
    assert enc.adapters == lora.text_state(rows) and len(enc.adapters) == 64
    after = api.encode_text(pipe, "ab")
    assert after.key != before.key and after is not before
    assert rel_l2(np.array(after.features, np.float64), want[0]) <= TEXT_TOL
    # The same state again is a no-op.
    assert lora.apply_text_adapters(pipe, rows) == 0
    assert lora.text_state([(a1, 0.8)]) != lora.text_state(rows)


def test_text_halves_come_off_and_change(tmp_path):
    """An empty list, a strength of 0 and a new list each set the encoder
    to exactly that state, with the conditioning key following."""
    one, two = _text_pairs(1, 6.0), _text_pairs(2)
    pipe = _pipe()
    enc = pipe.encoder
    a1 = lora.load_adapter(pipe, _write(tmp_path / "one.safetensors", one))
    a2 = lora.load_adapter(pipe, _write(tmp_path / "two.safetensors", two))
    base = np.array(enc.encode("ab").hidden[0])
    before = api.encode_text(pipe, "ab")

    def bare() -> None:
        assert _wrapped(enc.model) == [] and enc.adapters == ""
        assert np.array_equal(np.array(enc.encode("ab").hidden[0]), base)
        assert api.encode_text(pipe, "ab") is before

    for off in ([], [(a1, 0.0)], [(a1, 0.0), (a2, 0.0)]):
        assert lora.apply_text_adapters(pipe, [(a1, 0.8)]) == 1
        adapted = api.encode_text(pipe, "ab")
        assert adapted.key != before.key and enc.adapters
        assert lora.apply_text_adapters(pipe, off) == 0
        bare()
    # Another list replaces the one in place, with no trace of it.
    assert lora.apply_text_adapters(pipe, [(a1, 0.8)]) == 1
    first = api.encode_text(pipe, "ab")
    assert lora.apply_text_adapters(pipe, [(a1, 0.5), (a2, -0.5)]) == 2
    assert enc.adapters == lora.text_state([(a1, 0.5), (a2, -0.5)])
    got = np.array(enc.encode("ab").hidden[0], np.float64)
    assert rel_l2(got, _merged_hidden([(one, 0.5), (two, -0.5)])) <= TEXT_TOL
    second = api.encode_text(pipe, "ab")
    assert len({before.key, first.key, second.key}) == 3
    # A list that does not fit changes nothing.
    rng = np.random.default_rng(0)
    bad = lora.load_adapter(pipe, _write(
        tmp_path / "bad.safetensors", {Q: make_pair(rng, 2560, 2560, 2)}))
    with pytest.raises(lora.AdapterError, match="bad.safetensors"):
        lora.apply_text_adapters(pipe, [(a1, 0.5), (bad, 1.0)])
    assert enc.adapters == lora.text_state([(a1, 0.5), (a2, -0.5)])
    assert rel_l2(np.array(enc.encode("ab").hidden[0], np.float64),
                  _merged_hidden([(one, 0.5), (two, -0.5)])) <= TEXT_TOL
    assert lora.apply_text_adapters(pipe, []) == 0
    bare()
    # A caller's own encoder object that keeps no adapter state has
    # nothing to take off.
    from types import SimpleNamespace

    pipe.encoder = SimpleNamespace(owned=True, released=False)
    assert lora.apply_text_adapters(pipe, []) == 0
    assert lora.apply_text_adapters(pipe, [(a1, 0.0)]) == 0


def test_a_borrowed_encoder_refuses_text_halves_by_name(tmp_path):
    pipe = _pipe(owned=False)
    enc = pipe.encoder
    base = np.array(enc.encode("ab").hidden[0])
    ad = lora.load_adapter(pipe, _write(tmp_path / "style.safetensors",
                                        _text_pairs(1),
                                        tiny_z_adapter(1, 4.0)))
    with pytest.raises(lora.AdapterError,
                       match="style.safetensors: 3 text encoder pairs are "
                             "refused, because the text encoder is a "
                             "resident chat model") as e:
        lora.apply_text_adapters(pipe, [(ad, 1.0)])
    assert "text strength of 0" in str(e.value)
    # A text strength of zero applies the image half alone.
    assert lora.apply_text_adapters(pipe, [(ad, 0.0)]) == 0
    assert enc.adapters == "" and _wrapped(enc.model) == []
    assert np.array_equal(np.array(enc.encode("ab").hidden[0]), base)
    tid = api.define_patch_table(pipe, lora.table_entries([(ad, 1.0)]))
    assert tid != api.EMPTY_TABLE
    # An adapter with no text half asks nothing of the encoder.
    dit_only = lora.load_adapter(pipe, _write(
        tmp_path / "dit.safetensors", {}, tiny_z_adapter(2, 4.0)))
    assert lora.apply_text_adapters(pipe, [(dit_only, 1.0)]) == 0


def test_text_halves_need_an_encoder_and_a_fit(tmp_path):
    pipe = _pipe()
    rng = np.random.default_rng(0)
    bad = lora.load_adapter(pipe, _write(
        tmp_path / "bad.safetensors", {Q: make_pair(rng, 2560, 2560, 2)}))
    with pytest.raises(lora.AdapterError,
                       match="bad.safetensors: the text encoder pair on "
                             f"{Q} does not fit the tiny encoder: the pair "
                             "is 2560x2560"):
        lora.apply_text_adapters(pipe, [(bad, 1.0)])
    deep = lora.load_adapter(pipe, _write(
        tmp_path / "deep.safetensors",
        {"model.layers.30.mlp.up_proj": make_pair(rng, 96, 64, 2)}))
    with pytest.raises(lora.AdapterError, match="no such linear layer"):
        lora.apply_text_adapters(pipe, [(deep, 1.0)])
    assert _wrapped(pipe.encoder.model) == [] and pipe.encoder.adapters == ""
    good = lora.load_adapter(pipe, _write(tmp_path / "good.safetensors",
                                          _text_pairs(1)))
    pipe.encoder.release()
    with pytest.raises(lora.AdapterError, match="need a loaded text encoder"):
        lora.apply_text_adapters(pipe, [(good, 1.0)])
    pipe.encoder = None
    with pytest.raises(lora.AdapterError, match="need a loaded text encoder"):
        lora.apply_text_adapters(pipe, [(good, 1.0)])


def test_a_failed_text_install_leaves_the_encoder_bare(tmp_path,
                                                       monkeypatch):
    """A row that does not fit, or an install that fails part way, leaves
    no wrap and no adapter state, and the same call works afterwards."""
    import gmlx.load.modules as modules

    pipe = _pipe()
    enc = pipe.encoder
    rng = np.random.default_rng(0)
    one, two = _text_pairs(1), _text_pairs(2)
    good = lora.load_adapter(pipe, _write(tmp_path / "good.safetensors",
                                          one))
    also = lora.load_adapter(pipe, _write(tmp_path / "also.safetensors",
                                          two))
    bad = lora.load_adapter(pipe, _write(
        tmp_path / "bad.safetensors", {Q: make_pair(rng, 2560, 2560, 2)}))
    base = np.array(enc.encode("ab").hidden[0])
    before = api.encode_text(pipe, "ab")

    def bare() -> None:
        assert _wrapped(enc.model) == [] and enc.adapters == ""
        assert np.array_equal(np.array(enc.encode("ab").hidden[0]), base)
        assert api.encode_text(pipe, "ab") is before

    with pytest.raises(lora.AdapterError, match="bad.safetensors"):
        lora.apply_text_adapters(pipe, [(good, 1.0), (bad, 1.0)])
    bare()
    # An install that raises on the second adapter takes the first off.
    real, calls = modules.install_lora_adapter, []

    def flaky(model, text_plan, **kw):
        calls.append(kw["slot"])
        if len(calls) == 2:
            raise ValueError("no room")
        return real(model, text_plan, **kw)

    monkeypatch.setattr(modules, "install_lora_adapter", flaky)
    with pytest.raises(lora.AdapterError, match="also.safetensors.*no room"):
        lora.apply_text_adapters(pipe, [(good, 1.0), (also, 1.0)])
    assert calls == [0, 1]
    bare()
    monkeypatch.setattr(modules, "install_lora_adapter", real)
    assert lora.apply_text_adapters(pipe, [(good, 1.0), (also, 1.0)]) == 2
    got = np.array(enc.encode("ab").hidden[0], np.float64)
    assert rel_l2(got, _merged_hidden([(one, 1.0), (two, 1.0)])) <= TEXT_TOL
