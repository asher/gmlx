"""The H3 encoder spec's token ids on a GGUF tokenizer laid out as the
stock Qwen3-VL-32B file's: filler entries at the seven ids MiniMax's
tokenizer adds. CPU-only, no weights."""
from __future__ import annotations

import dataclasses

import pytest
from video_fixtures import gguf_tokenizer, tokenizer_kv

from gmlx.image import text_encoder as te
from gmlx.video import encoder, memfit
from gmlx.video.families.h3 import family as hf

SPEC = hf.FAMILY.encoder()


@pytest.fixture(scope="module")
def tok():
    return gguf_tokenizer()


def _enc(tok, text):
    return tok.encode(text, add_special_tokens=False)


def test_the_seven_tokens_take_their_fixed_ids_between_the_pieces(tok):
    prompt = ("He says <d>[English] Hello.</d> then sings "
              "<|lyrics_start|>la la<|lyrics_end|><|cutoff|>")
    assert te.token_ids(SPEC, tok, prompt) == (
        _enc(tok, "He says ") + [151669] + _enc(tok, "[English] Hello.")
        + [151670] + _enc(tok, " then sings ") + [151672]
        + _enc(tok, "la la") + [151673, 151671])
    for text, i in hf.ADDED_TOKENS:
        assert tok.convert_ids_to_tokens(i) == f"[PAD{i}]"
        assert i not in _enc(tok, text)
        assert te.token_ids(SPEC, tok, text) == [i]


def test_a_prompt_without_the_tokens_is_the_tokenizers_own(tok):
    text = "He walks in the rain, umbrella up"
    assert te.token_ids(SPEC, tok, text) == _enc(tok, text)


def test_an_empty_prompt_is_the_pad_token(tok):
    assert te.token_ids(SPEC, tok, "") == [151643]
    no_empty = dataclasses.replace(SPEC, empty_ids=())
    with pytest.raises(ValueError, match="has no tokens"):
        te.token_ids(no_empty, tok, "")


@pytest.mark.parametrize("prompt", [
    "", "He", "He says <d>[English] Hello.</d>", "<|cutoff|>",
    "cafe\u0301 au lait", "a cat " + "\ufb2c" * 12])
def test_a_plan_counts_the_prompts_own_rows(tok, prompt):
    """The ids the encoder reads, the seven fixed ids, the NFC form and
    the empty prompt among them."""
    assert memfit.prompt_rows(hf.FAMILY, prompt, tok) == len(
        te.token_ids(SPEC, tok, prompt))


def _gguf(path, arch="qwen3vl", hidden=5120, layers=64, tokens=True):
    """A GGUF with the tokenizer metadata of ``tokenizer_kv`` and no
    tensors."""
    import gguf

    kv = tokenizer_kv()
    w = gguf.GGUFWriter(str(path), arch=arch)
    if hidden is not None:
        w.add_embedding_length(hidden)
        w.add_block_count(layers)
    if tokens:
        w.add_tokenizer_model(kv["tokenizer.ggml.model"])
        w.add_tokenizer_pre(kv["tokenizer.ggml.pre"])
        w.add_token_list(kv["tokenizer.ggml.tokens"])
        w.add_token_merges(kv["tokenizer.ggml.merges"])
        w.add_token_types(kv["tokenizer.ggml.token_type"])
        w.add_eos_token_id(kv["tokenizer.ggml.eos_token_id"])
        w.add_pad_token_id(kv["tokenizer.ggml.padding_token_id"])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return str(path)


def test_the_header_tokenizer_is_the_ggufs_own(tmp_path, tok, capsys):
    got = encoder.header_tokenizer(hf.FAMILY, _gguf(tmp_path / "enc.gguf"))
    for p in ("He says <d>[English] Hello.</d> then", "", "\ufb2c" * 3):
        assert te.token_ids(SPEC, got, p) == te.token_ids(SPEC, tok, p)
    # Built quietly: generate and bench print only their own lines.
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("kw, why", [
    ({"arch": "clip", "hidden": None, "tokens": False},
     "arch 'clip', expected 'qwen3', hidden size None, expected 5120, "
     "None layers, expected 64"),
    ({"arch": "", "hidden": None, "tokens": False},
     "arch None, expected 'qwen3', hidden size None, expected 5120, "
     "None layers, expected 64"),
    ({"layers": 50}, "50 layers, expected 64"),
    ({"hidden": 4096, "layers": 36}, "hidden size 4096, expected 5120, "
     "36 layers, expected 64")])
def test_a_file_that_is_not_the_encoder_is_refused_before_pricing(
        tmp_path, kw, why):
    """An mmproj, a file with no metadata (the DiT, the H3 encoder file),
    a cut-down Qwen3-VL-32B and a smaller Qwen3-VL are refused by the
    load's own message, before their tokenizer is read."""
    path = _gguf(tmp_path / "x.gguf", **kw)
    with pytest.raises(te.EncoderMismatchError) as ei:
        encoder.header_tokenizer(hf.FAMILY, path)
    assert str(ei.value) == f"the text encoder is not {SPEC.name}: {why}"
