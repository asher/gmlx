"""The header check for a token embedding adapter: ``embedding_refusal``
and ``adapter_load_refusal`` read the base and adapter headers and refuse,
before any build, a pair the base cannot take. CPU-only (minted GGUF
headers, no model)."""
from __future__ import annotations

import pytest

gguf = pytest.importorskip("gguf")
from gmlx.load import adapter  # noqa: E402

from lora_gguf import D, R, V, mint_adapter, mint_base  # noqa: E402


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
