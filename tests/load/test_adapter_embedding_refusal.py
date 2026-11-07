"""The header check for a token embedding adapter: ``embedding_refusal``
and ``adapter_load_refusal`` read the base and adapter headers and refuse,
before any build, a pair the base cannot take. CPU-only (minted GGUF
headers, no model)."""
from __future__ import annotations

import numpy as np
import pytest

gguf = pytest.importorskip("gguf")
from gmlx.load import adapter  # noqa: E402

V, D, R = 32, 8, 2


def _write(path, arch, tensors, kv=()):
    w = gguf.GGUFWriter(str(path), arch)
    for key, kind, value in kv:
        getattr(w, f"add_{kind}")(key, value)
    for name, arr in tensors.items():
        w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def mint_base(path, *, arch="llama", tied=False, vocab=V, kv=()):
    tensors = {"token_embd.weight": np.zeros((vocab, D), np.float32),
               "blk.0.attn_q.weight": np.zeros((D, D), np.float32)}
    if not tied:
        tensors["output.weight"] = np.zeros((vocab, D), np.float32)
    return _write(path, arch, tensors, kv)


def mint_adapter(path, *, arch="llama", embd=True, vocab=V):
    tensors = {"blk.0.attn_q.weight.lora_a": np.zeros((R, D), np.float32),
               "blk.0.attn_q.weight.lora_b": np.zeros((D, R), np.float32)}
    if embd:
        tensors["token_embd.weight.lora_a"] = np.zeros((vocab, R), np.float32)
        tensors["token_embd.weight.lora_b"] = np.zeros((D, R), np.float32)
    kv = (("adapter.type", "string", "lora"), ("adapter.lora.alpha", "float32", 2.0))
    return _write(path, arch, tensors, kv)


def test_an_untied_base_takes_the_pair(tmp_path):
    base = mint_base(tmp_path / "base.gguf")
    assert adapter.embedding_refusal(mint_adapter(tmp_path / "a.gguf"), base) is None


def test_an_adapter_without_an_embedding_pair_passes_a_tied_base(tmp_path):
    base = mint_base(tmp_path / "base.gguf", tied=True)
    assert adapter.embedding_refusal(mint_adapter(tmp_path / "a.gguf", embd=False), base) is None


def test_a_tied_base_is_refused(tmp_path):
    base = mint_base(tmp_path / "base.gguf", tied=True)
    why = adapter.embedding_refusal(mint_adapter(tmp_path / "a.gguf"), base)
    assert why == "a.gguf: " + adapter._EMBD_TIED


def test_a_pair_of_another_vocabulary_is_refused(tmp_path):
    base = mint_base(tmp_path / "base.gguf")
    why = adapter.embedding_refusal(mint_adapter(tmp_path / "a.gguf", vocab=V + 4), base)
    assert why == "a.gguf: " + adapter.embd_shape_message((V + 4, R), (D, R), V, D)


def test_granite_switch_is_refused(tmp_path):
    base = mint_base(tmp_path / "base.gguf", arch="graniteswitch")
    why = adapter.embedding_refusal(mint_adapter(tmp_path / "a.gguf", arch="graniteswitch"), base)
    assert why == "a.gguf: " + adapter._EMBD_GRANITE_SWITCH


def test_a_hadamard_folded_table_is_refused(tmp_path):
    kv = (("prism.hadamard.version", "uint32", 1),
          ("prism.hadamard.weight_names", "array", ["token_embd.weight"]))
    base = mint_base(tmp_path / "base.gguf", kv=kv)
    why = adapter.embedding_refusal(mint_adapter(tmp_path / "a.gguf"), base)
    assert why == "a.gguf: " + adapter._EMBD_FOLDED


def test_the_load_refusal_raises_the_first_refusal_over_its_adapters(tmp_path):
    base = mint_base(tmp_path / "base.gguf", tied=True)
    ok = mint_adapter(tmp_path / "ok.gguf", embd=False)
    bad = mint_adapter(tmp_path / "bad.gguf")
    assert adapter.adapter_load_refusal(base, []) is None
    assert adapter.adapter_load_refusal(base, [ok]) is None
    with pytest.raises(ValueError) as e:
        adapter.adapter_load_refusal(base, [ok, bad])
    assert str(e.value) == "bad.gguf: " + adapter._EMBD_TIED
