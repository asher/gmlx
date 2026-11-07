"""token_bytes, whitespace_start_mask and vocab_map_hash on two minted
tokenizers, one ByteLevel BPE and one SentencePiece-style BPE with byte
fallback: every encoding concatenates back to the source bytes, the
whitespace mask marks space-initial ids and the EOS, and the hash tells the
two vocabularies apart. CPU, no model."""
from __future__ import annotations

import pytest

from gmlx.load.tokenizer import token_bytes, vocab_map_hash, whitespace_start_mask

SPM_SPACE = "\u2581"


def _bytelevel_tokenizer(extra_merges):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    from gmlx._vendor.mlx_lm.tokenizer_utils import BPEStreamingDetokenizer

    BPEStreamingDetokenizer.make_byte_decoder()
    chars = sorted(BPEStreamingDetokenizer._byte_decoder.items(), key=lambda kv: kv[1])
    vocab = {ch: i for i, (ch, _b) in enumerate(chars)}
    merges = []
    for a, b in extra_merges:
        merged = a + b
        if merged not in vocab:
            vocab[merged] = len(vocab)
        merges.append((a, b))
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>", bos_token="<bos>")


def _spm_tokenizer(pieces, merges):
    from tokenizers import Tokenizer, decoders, models, normalizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2}
    for i in range(256):
        vocab[f"<0x{i:02X}>"] = len(vocab)
    for p in pieces:
        if p not in vocab:
            vocab[p] = len(vocab)
    for a, b in merges:
        for piece in (a, b, a + b):
            if piece not in vocab:
                vocab[piece] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, byte_fallback=True, unk_token=None))
    tok.normalizer = normalizers.Replace(" ", SPM_SPACE)
    tok.decoder = decoders.Sequence([decoders.Replace(SPM_SPACE, " "), decoders.ByteFallback()])
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>", bos_token="<bos>",
                                   pad_token="<pad>")


@pytest.fixture(scope="module")
def tok_bl():
    return _bytelevel_tokenizer([("\u0120", "t"), ("h", "e"), ("\u0120t", "he"), ("1", "2"),
                                 ("12", "3"), ("\u0120", "a"), ("i", "s"), ("\u0120", "is"),
                                 ("c", "a"), ("ca", "t")])


@pytest.fixture(scope="module")
def tok_spm():
    import string
    pieces = ([SPM_SPACE] + list(string.ascii_letters + string.digits + ".,!?\n")
              + [SPM_SPACE + c for c in string.ascii_lowercase]
              + [SPM_SPACE + w for w in ("the", "cat", "is", "1")]
              + ["th", "he", "at", "12", "123"])
    merges = [(SPM_SPACE, "t"), (SPM_SPACE + "t", "h"), (SPM_SPACE + "th", "e"),
              ("t", "h"), ("h", "e"), ("a", "t"),
              (SPM_SPACE, "c"), (SPM_SPACE + "c", "a"), (SPM_SPACE + "ca", "t"),
              (SPM_SPACE, "i"), (SPM_SPACE + "i", "s"),
              ("1", "2"), ("12", "3"), (SPM_SPACE, "1")]
    return _spm_tokenizer(pieces, merges)


def _ids(tok, text):
    return tok._tokenizer.encode(text, add_special_tokens=False).ids


def test_token_bytes_concatenate_to_the_source_bytes(tok_bl, tok_spm):
    text = "the cat is 123 the\u00e9 \u4e2d\u6587 !!"
    for tok in (tok_bl, tok_spm):
        tb = token_bytes(tok)
        assert b"".join(tb[i] for i in _ids(tok, text)) == text.encode("utf-8")


def test_specials_and_padded_head_ids_are_none(tok_spm):
    tb = token_bytes(tok_spm, width=len(tok_spm) + 5)
    assert len(tb) == len(tok_spm) + 5
    assert tb[-5:] == [None] * 5
    assert tb[tok_spm.eos_token_id] is None
    assert tb[tok_spm.pad_token_id] is None


def test_whitespace_mask_marks_space_initial_ids_and_eos(tok_bl, tok_spm):
    for tok in (tok_bl, tok_spm):
        tb = token_bytes(tok)
        m = whitespace_start_mask(tok, len(tb), tb)
        assert m[_ids(tok, " the")[0]]
        assert not m[_ids(tok, "cat")[0]]
        assert m[tok.eos_token_id]


def test_vocab_map_hash_is_stable_and_separates_vocabularies(tok_bl, tok_spm):
    assert vocab_map_hash(tok_bl) == vocab_map_hash(tok_bl)
    assert vocab_map_hash(tok_bl) != vocab_map_hash(tok_spm)
    assert len(vocab_map_hash(tok_bl)) == 16


def test_token_bytes_for_matches_token_bytes_on_every_plain_id(tok_bl, tok_spm):
    from gmlx.load.tokenizer import special_ids, token_bytes_for

    for tok in (tok_bl, tok_spm):
        tb = token_bytes(tok)
        special = special_ids(tok)
        ids = [i for i in range(len(tb)) if i not in special]
        assert token_bytes_for(tok, ids) == [tb[i] for i in ids]


def test_byte_fallback_pieces_keep_their_own_byte(tok_spm):
    from gmlx.load.tokenizer import token_bytes_for

    ids = tok_spm.convert_tokens_to_ids(["<0xE4>", "<0xB8>", "<0xAD>"])
    assert token_bytes_for(tok_spm, ids) == [b"\xe4", b"\xb8", b"\xad"]


def test_specials_get_their_text_and_padded_ids_get_nothing(tok_spm):
    from gmlx.load.tokenizer import token_bytes_for

    got = token_bytes_for(tok_spm, [tok_spm.eos_token_id, len(tok_spm) + 3])
    assert got == [b"<eos>", b""]


def test_a_tokenizer_without_a_fast_backend_gets_decoded_bytes(caplog):
    from gmlx.load.tokenizer import token_bytes_for

    class Slow:
        def decode(self, ids):
            return {7: "é", 8: "x"}[ids[0]]

        def convert_ids_to_tokens(self, ids):
            raise AssertionError("no vocabulary lookup in fallback mode")

    slow = Slow()
    with caplog.at_level("WARNING", logger="gmlx.load.tokenizer"):
        assert token_bytes_for(slow, [7, 8]) == ["é".encode(), b"x"]
        assert token_bytes_for(slow, [8]) == [b"x"]
    assert sum("fast backend" in r.message for r in caplog.records) == 1


def test_the_context_is_built_under_the_lock(tok_bl):
    from gmlx.load import tokenizer as tk

    tok = tk.PreTrainedTokenizerFast(tokenizer_object=tok_bl._tokenizer)
    held = []

    class Lock:
        def __enter__(self):
            held.append(True)

        def __exit__(self, *exc):
            return False

    tk.token_bytes_for(tok, [5], lock=Lock())
    tk.token_bytes_for(tok, [6], lock=Lock())
    assert held == [True]
