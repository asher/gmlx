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


@pytest.mark.parametrize("kw, a_shape, b_shape", [
    ({"vocab": V + 4}, (V + 4, R), (D, R)),
    ({"n_embd": D + 4}, (V, R), (D + 4, R)),
    ({"b_rank": R + 1}, (V, R), (D, R + 1)),
    ({"b": False}, (V, R), ()),
], ids=["vocab", "n_embd", "rank", "no-b"])
def test_a_pair_that_does_not_fit_the_base_is_refused(tmp_path, kw, a_shape, b_shape):
    base = mint_base(tmp_path / "base.gguf")
    why = adapter.embedding_refusal(mint_adapter(tmp_path / "a.gguf", **kw), base)
    assert why == "a.gguf: " + adapter.embd_shape_message(a_shape, b_shape, V, D)


def test_granite_switch_is_refused(tmp_path):
    base = mint_base(tmp_path / "base.gguf", arch="graniteswitch")
    why = adapter.embedding_refusal(mint_adapter(tmp_path / "a.gguf", arch="graniteswitch"), base)
    assert why == "a.gguf: " + adapter._EMBD_GRANITE_SWITCH


_FOLD_KV = (("prism.hadamard.version", "uint32", 1),
            ("prism.hadamard.block_size", "uint32", 4),
            ("prism.hadamard.transform", "string", "normalized-sylvester-walsh-hadamard"),
            ("prism.hadamard.axis", "string", "input-last-dimension"),
            ("prism.hadamard.sign_mode", "string", "identity"),
            ("prism.hadamard.weight_names", "array", ["blk.0.attn_q.weight"]))


def _spec_of(path):
    from gmlx.load.hadamard import parse_hadamard_spec
    from gmlx.load.headerscan import scan_gguf

    return parse_hadamard_spec(scan_gguf(path, include_tensors=False).kv)


def test_a_hadamard_folded_table_is_refused(tmp_path):
    kv = _FOLD_KV + (("prism.hadamard.inverse_weight_names", "array", ["token_embd.weight"]),)
    base = mint_base(tmp_path / "base.gguf", kv=kv)
    assert _spec_of(base).inverse_names == ("token_embd.weight",)
    why = adapter.embedding_refusal(mint_adapter(tmp_path / "a.gguf"), base)
    assert why == "a.gguf: " + adapter._EMBD_FOLDED


def test_a_fold_that_leaves_the_table_alone_takes_the_pair(tmp_path):
    base = mint_base(tmp_path / "base.gguf", kv=_FOLD_KV)
    assert _spec_of(base).inverse_names == ()
    assert adapter.embedding_refusal(mint_adapter(tmp_path / "a.gguf"), base) is None


def test_a_per_layer_table_pair_is_refused_from_the_headers(tmp_path):
    base = mint_base(tmp_path / "base.gguf", tied=True)
    path = mint_adapter(tmp_path / "a.gguf", embd=False, per_layer=True)
    why = adapter.embedding_refusal(path, base)
    assert why == "a.gguf: " + adapter.per_layer_embd_message("per_layer_token_embd.weight")


def test_the_load_refusal_raises_the_first_refusal_over_its_adapters(tmp_path):
    base = mint_base(tmp_path / "base.gguf", tied=True)
    ok = mint_adapter(tmp_path / "ok.gguf", embd=False)
    bad = mint_adapter(tmp_path / "bad.gguf")
    assert adapter.adapter_load_refusal(base, []) is None
    assert adapter.adapter_load_refusal(base, [ok]) is None
    with pytest.raises(ValueError) as e:
        adapter.adapter_load_refusal(base, [ok, bad])
    assert str(e.value) == "bad.gguf: " + adapter._EMBD_TIED


@pytest.mark.parametrize("data", [b"GGUF", b"GGUF\x03\x00\x00\x00", b"not a gguf"])
def test_an_adapter_that_is_not_a_gguf_is_left_to_the_load(tmp_path, data):
    base = mint_base(tmp_path / "base.gguf")
    bad = tmp_path / "a.gguf"
    bad.write_bytes(data)
    assert adapter.adapter_load_refusal(str(base), [str(bad)]) is None
